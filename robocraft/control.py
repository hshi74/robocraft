import glob
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

import taichi as ti
ti.init(arch=ti.cpu)

import torch

from config import gen_args
from matplotlib import cm
from model import Model, EarthMoverLoss, ChamferLoss, HausdorffLoss
from plb.engine.taichi_env import TaichiEnv
from plb.config import load
from plb.algorithms import sample_data
from tqdm import tqdm, trange
from transforms3d.quaternions import *
from transforms3d.axangles import axangle2mat
from sys import platform
from utils import set_seed,  Tee, count_parameters
from utils import load_data, get_scene_info, get_env_group, prepare_input


use_gpu = True
device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")


task_params = {
    "mid_point": np.array([0.5, 0.14, 0.5, 0, 0, 0]),
    "sample_radius": 0.4,
    "len_per_grip": 30,
    "len_per_grip_back": 10,
    "floor_pos": np.array([0.5, 0, 0.5]),
    "n_shapes": 3, 
    "n_shapes_floor": 9,
    "n_shapes_per_gripper": 11,
    "gripper_mid_pt": int((11 - 1) / 2),
    "gripper_gap_limits": np.array([0.14, 0.06]), # ((0.4 * 2 - (0.23)) / (2 * 30), (0.4 * 2 - 0.15) / (2 * 30)),
    "p_noise_scale": 0.08,
    "p_noise_bound": 0.1,
    "loss_weights": [0.9, 0.1, 0.0],
    "tool_size": 0.045,
    "CEM_opt_iter": 3,
    "CEM_init_pose_sample_size": 40,
	"CEM_gripper_rate_sample_size": 8,
    "GD_batch_size": 1
}

emd_loss = EarthMoverLoss()
chamfer_loss = ChamferLoss()
h_loss = HausdorffLoss()


def visualize_sampled_init_pos(init_pose_seqs, reward_seqs, idx, path):
    init_pose_seqs = init_pose_seqs.cpu().numpy()
    reward_seqs = reward_seqs.cpu().numpy()
    idx = idx.cpu().numpy()

    n_subplots = init_pose_seqs.shape[1]
    fig, axs = plt.subplots(1, n_subplots)
    fig.set_size_inches(8 * n_subplots, 4.8)
    for i in range(n_subplots):
        if n_subplots == 1:
            ax = axs
        else:
            ax = axs[i]
        ax.scatter(init_pose_seqs[:, i, task_params["gripper_mid_pt"], 0], 
            init_pose_seqs[:, i, task_params["gripper_mid_pt"], 2],
            c=reward_seqs, cmap=cm.jet)

        ax.set_title(f"GRIP {i+1}")
        ax.set_xlabel('x coordinate')
        ax.set_ylabel('z coordinate')

    color_map = cm.ScalarMappable(cmap=cm.jet)
    color_map.set_array(reward_seqs[idx])
    plt.colorbar(color_map, ax=axs)

    plt.savefig(path)
    # plt.show()


def visualize_loss(loss_lists, path):
    plt.figure(figsize=[16, 9])
    for loss_list in loss_lists:
        iters, loss = map(list, zip(*loss_list))
        plt.plot(iters, loss, linewidth=6)
    plt.xlabel('epochs', fontsize=30)
    plt.ylabel('loss', fontsize=30)
    plt.title('Test Loss', fontsize=35)
    # plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    plt.savefig(path)
    # plt.show()


def visualize_rollout_loss(loss_lists, path):
    plt.figure(figsize=[16, 9])
    labels = ["model", "sim"]
    for label, loss_list in zip(labels, loss_lists):
        iters, loss = map(list, zip(*loss_list))
        plt.plot(iters, loss, label=label, linewidth=6)
    plt.xlabel('iteration', fontsize=30)
    plt.ylabel('loss', fontsize=30)
    plt.title('Rollout Loss', fontsize=35)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    plt.savefig(path)
    # plt.show()


def visualize_points(all_points, n_particles, path):
    # print(all_points.shape)
    points = all_points[:n_particles]
    shapes = all_points[n_particles:]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45, 135)
    ax.scatter(points[:, 0], points[:, 2], points[:, 1], c='b', s=20)
    ax.scatter(shapes[:, 0], shapes[:, 2], shapes[:, 1], c='r', s=20)
    
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = 0.25  # maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    ax.invert_yaxis()

    plt.savefig(path)
    # plt.show()


def visualize_points_helper(ax, all_points, n_particles, p_color='b'):
    points = ax.scatter(all_points[:n_particles, 0], all_points[:n_particles, 2], all_points[:n_particles, 1], c=p_color, s=10)
    shapes = ax.scatter(all_points[n_particles:, 0], all_points[n_particles:, 2], all_points[n_particles:, 1], c='r', s=10)

    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = 0.25  # maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    ax.invert_yaxis()

    return points, shapes


def plt_render(particles_set, target_shape, n_particle, render_path):
    # particles_set[0] = np.concatenate((particles_set[0][:, :n_particle], particles_set[1][:, n_particle:]), axis=1)
    n_frames = particles_set[0].shape[0]
    rows = 2
    cols = 3

    fig, big_axes = plt.subplots(rows, 1, figsize=(3*cols, 3*rows))
    row_titles = ['Simulator', 'Model']
    views = [(90, 90), (0, 90), (45, 135)]
    plot_info_all = {}
    for i in range(rows):
        big_axes[i].set_title(row_titles[i], fontweight='semibold')
        big_axes[i].axis('off')

        plot_info = []
        for j in range(cols):
            ax = fig.add_subplot(rows, cols, i * cols + j + 1, projection='3d')
            ax.view_init(*views[j])
            visualize_points_helper(ax, target_shape, n_particle, p_color='c')
            points, shapes = visualize_points_helper(ax, particles_set[i][0], n_particle)
            plot_info.append((points, shapes))

        plot_info_all[row_titles[i]] = plot_info

    plt.tight_layout()
    # plt.show()

    def update(step):
        outputs = []
        for i in range(rows):
            states = particles_set[i]
            for j in range(cols):
                points, shapes = plot_info_all[row_titles[i]][j]
                points._offsets3d = (states[step, :n_particle, 0], states[step, :n_particle, 2], states[step, :n_particle, 1])
                shapes._offsets3d = (states[step, n_particle:, 0], states[step, n_particle:, 2], states[step, n_particle:, 1])
                outputs.append(points)
                outputs.append(shapes)
        return outputs

    anim = animation.FuncAnimation(fig, update, frames=np.arange(0, n_frames), blit=False)

    # plt.show()
    anim.save(render_path, writer=animation.PillowWriter(fps=20))


def expand(batch_size, info):
    length = len(info.shape)
    if length == 2:
        info = info.expand([batch_size, -1])
    elif length == 3:
        info = info.expand([batch_size, -1, -1])
    elif length == 4:
        info = info.expand([batch_size, -1, -1, -1])
    return info


def random_rotate(mid_point, z_vec, z_angle):
    mid_point = mid_point[:3]
    z_mat = axangle2mat(z_vec, z_angle, is_normalized=True)

    all_mat = z_mat
    quat = torch.tensor(mat2quat(all_mat))
    
    return quat


def get_pose(new_mid_point, rot_noise, z_angle, mode):
    # if not torch.is_tensor(new_mid_point):
    #     new_mid_point = torch.tensor(new_mid_point)
    
    if not torch.is_tensor(rot_noise):
        rot_noise = torch.tensor(rot_noise)

    x1 = new_mid_point[0] - task_params["sample_radius"] * torch.cos(rot_noise)
    y1 = new_mid_point[2] + task_params["sample_radius"] * torch.sin(rot_noise)
    x2 = new_mid_point[0] + task_params["sample_radius"] * torch.cos(rot_noise)
    y2 = new_mid_point[2] - task_params["sample_radius"] * torch.sin(rot_noise)
    
    unit_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
    
    gripper1_pos = torch.tensor([x1, new_mid_point[1], y1])
    gripper2_pos = torch.tensor([x2, new_mid_point[1], y2])
    
    if mode == '3d':
        z_vec = torch.tensor([torch.cos(rot_noise), 0, torch.sin(rot_noise)])
        unit_quat = random_rotate(new_mid_point, z_vec, z_angle)
    
    # import pdb; pdb.set_trace()
    new_prim1 = []
    for j in range(task_params["n_shapes_per_gripper"]):
        prim1_pos = torch.stack([x1, torch.tensor(new_mid_point[1].item() + 0.018 * (j-5)), y1])
        prim1_tmp = torch.cat((prim1_pos, unit_quat))
        new_prim1.append(prim1_tmp)
    new_prim1 = torch.stack(new_prim1)
    
    # import pdb; pdb.set_trace()
    if mode == '3d':
        new_prim1_pos = (torch.tensor(quat2mat(unit_quat)) @ (new_prim1[:, :3] - gripper1_pos).T).T + gripper1_pos
        new_prim1 = torch.cat((new_prim1_pos, new_prim1[:, 3:]), 1)
    
    new_prim2 = []
    for j in range(task_params["n_shapes_per_gripper"]):
        prim2_pos = torch.stack([x2, torch.tensor(new_mid_point[1].item() + 0.018 * (j-5)), y2])
        prim2_tmp = torch.cat((prim2_pos, unit_quat))
        new_prim2.append(prim2_tmp)
    new_prim2 = torch.stack(new_prim2)
    
    if mode == '3d':
        new_prim2_pos = (torch.tensor(quat2mat(unit_quat)) @ (new_prim2[:, :3] - gripper2_pos).T).T + gripper2_pos
        new_prim2 = torch.cat((new_prim2_pos, new_prim2[:, 3:]), 1)
    
    init_pose = torch.cat((new_prim1, new_prim2), 1)

    return init_pose


def get_action_seq(rot_noise, gripper_rate):
    if not torch.is_tensor(rot_noise):
        rot_noise = torch.tensor(rot_noise)
    
    if not torch.is_tensor(gripper_rate):
        gripper_rate = torch.tensor(gripper_rate)

    # n_actions = (gripper_rate - torch.remainder(gripper_rate, task_params["gripper_rate"])) / task_params["gripper_rate"]
    # n_actions = gripper_rate / task_params["gripper_rate"]
    zero_pad = torch.zeros(3)
    actions = []
    counter = 0
    while counter < task_params["len_per_grip"]:
        x = gripper_rate * torch.cos(rot_noise)
        y = -gripper_rate * torch.sin(rot_noise)
        prim1_act = torch.stack([x/0.02, torch.tensor(0), y/0.02])
        prim2_act = torch.stack([-x/0.02, torch.tensor(0), -y/0.02])
        act = torch.cat((prim1_act, zero_pad, prim2_act, zero_pad))
        actions.append(act)
        counter += 1

    # actions = actions[:task_params["len_per_grip"]]
    # for _ in range(task_params["len_per_grip"] - len(actions)):
    #     actions.append(torch.zeros(12))

    counter = 0
    while counter < task_params["len_per_grip_back"]:
        x = -gripper_rate * torch.cos(rot_noise)
        y = gripper_rate * torch.sin(rot_noise)
        prim1_act = torch.stack([x/0.02, torch.tensor(0), y/0.02])
        prim2_act = torch.stack([-x/0.02, torch.tensor(0), -y/0.02])
        act = torch.cat((prim1_act, zero_pad, prim2_act, zero_pad))
        actions.append(act)
        counter += 1

    actions = torch.stack(actions)
    # print(f"Sampled actions: {actions}")

    return actions


def get_params_from_pose(init_pose_seq):
    # import pdb; pdb.set_trace()
    if not torch.is_tensor(init_pose_seq):
        init_pose_seq = torch.tensor(init_pose_seq)
        
    mid_point_seq = (init_pose_seq[:, task_params["gripper_mid_pt"], :3] + init_pose_seq[:, task_params["gripper_mid_pt"], 7:10]) / 2

    angle_seq = torch.atan2(init_pose_seq[:, task_params["gripper_mid_pt"], 2] - mid_point_seq[:, 2], \
        init_pose_seq[:, task_params["gripper_mid_pt"], 0] - mid_point_seq[:, 0])

    a = init_pose_seq[:, 0, :3] - init_pose_seq[:, -1, :3]
    b = torch.tensor([[0.0, 1.0, 0.0]]).expand(init_pose_seq.shape[0], -1)
    z_angle_seq = torch.acos((a * b).sum(dim=1) / (a.pow(2).sum(dim=1).pow(0.5) * b.pow(2).sum(dim=1).pow(0.5)))

    pi = torch.full(angle_seq.shape, math.pi)
    angle_seq_new = pi - angle_seq
    z_angle_seq_new = pi - z_angle_seq
    
    return mid_point_seq, angle_seq_new, z_angle_seq_new


def get_action_seq_from_pose(init_pose_seq, gripper_rates):
    # import pdb; pdb.set_trace()
    _, rot_noise_seq, _ = get_params_from_pose(init_pose_seq)
    act_seq = []
    for i in range(len(rot_noise_seq)):
        act_seq.append(get_action_seq(rot_noise_seq[i], gripper_rates[i]))

    act_seq = torch.stack(act_seq)

    return act_seq


def get_gripper_rate_from_action_seq(act_seq):
    gripper_rate_seq = []
    for i in range(act_seq.shape[0]):
        gripper_rate = torch.linalg.norm(act_seq[i, 0, :3] * 0.02)
        gripper_rate_seq.append(gripper_rate)
    
    gripper_rate_seq = torch.stack(gripper_rate_seq)

    return gripper_rate_seq


def sample_particles(env, k_fps_particles, n_particles=2000):
    prim_pos1 = env.primitives.primitives[0].get_state(0)
    prim_pos2 = env.primitives.primitives[1].get_state(0)
    prim_pos = [prim_pos1[:3], prim_pos2[:3]]
    prim_rot = [prim_pos1[3:], prim_pos2[3:]]

    img = env.render_multi(mode='rgb_array', spp=3)
    rgb, depth = img[0], img[1]

    tool_info = {'tool_size': task_params["tool_size"]}

    ext1=env.renderer.get_ext(env.render_cfg.camera_rot_1, np.array(env.render_cfg.camera_pos_1))
    ext2=env.renderer.get_ext(env.render_cfg.camera_rot_2, np.array(env.render_cfg.camera_pos_2))
    ext3=env.renderer.get_ext(env.render_cfg.camera_rot_3, np.array(env.render_cfg.camera_pos_3))
    ext4=env.renderer.get_ext(env.render_cfg.camera_rot_4, np.array(env.render_cfg.camera_pos_4))
    intrinsic = env.renderer.get_int()
    cam_params = {'cam1_ext': ext1, 'cam2_ext': ext2, 'cam3_ext': ext3, 'cam4_ext': ext4, 'intrinsic': intrinsic}

    sampled_points = sample_data.gen_data_one_frame(rgb, depth, cam_params, prim_pos, prim_rot, tool_info, back=False, prev_pcd=[])

    positions = sample_data.update_position(task_params["n_shapes"], prim_pos, pts=sampled_points, 
                                            floor=task_params["floor_pos"])
    shape_positions = sample_data.shape_aug(positions, k_fps_particles)

    return shape_positions


def add_shapes(state_seq, init_pose_seq, act_seq, k_fps_particles, mode):
    updated_state_seq = []
    for i in range(act_seq.shape[0]):
        prim_pos1 = init_pose_seq[i, task_params["gripper_mid_pt"], :3].clone()
        prim_pos2 = init_pose_seq[i, task_params["gripper_mid_pt"], 7:10].clone()
        prim_rot1 = init_pose_seq[i, task_params["gripper_mid_pt"], 3:7].clone()
        prim_rot2 = init_pose_seq[i, task_params["gripper_mid_pt"], 10:].clone()
        for j in range(act_seq.shape[1]):
            idx = i * act_seq.shape[1] + j
            prim_pos1 += 0.02 * act_seq[i, j, :3]
            prim_pos2 += 0.02 * act_seq[i, j, 6:9]
            positions = sample_data.update_position(task_params["n_shapes"], [prim_pos1, prim_pos2], pts=state_seq[idx], 
                                                    floor=task_params["floor_pos"])
            if mode == '2d':
                shape_positions = sample_data.shape_aug(positions, k_fps_particles)
            else:
                shape_positions = sample_data.shape_aug_3D(positions, prim_rot1, prim_rot2, k_fps_particles)
            updated_state_seq.append(shape_positions)
    return np.stack(updated_state_seq)


class Planner(object):
    def __init__(self, args, taichi_env, env_init_state, scene_params, n_particle, n_shape, model, all_p, 
                goal_shapes, task_params, use_gpu, rollout_path, env="gripper"):
        self.args = args
        self.batch_size = args.shooting_batch_size
        self.sample_size = args.shooting_batch_size

        self.taichi_env = taichi_env
        self.env_init_state = env_init_state
        self.scene_params = scene_params
        self.n_particle = n_particle
        self.n_shape = n_shape
        self.model = model
        self.all_p = all_p
        self.goal_shapes = goal_shapes
        self.task_params = task_params
        self.use_gpu = use_gpu
        self.rollout_path = rollout_path
        self.env = env

        self.grip_cur = ''
        self.CEM_opt_iter_cur = 0

        if args.debug:
            self.batch_size = 1
            self.sample_size = 4
            task_params["CEM_init_pose_sample_size"] = 4
            task_params["CEM_gripper_rate_sample_size"] = 4


    def trajectory_optimization(self):
        state_goal_final = self.get_state_goal(self.args.n_grips - 1)
        visualize_points(state_goal_final[-1], self.n_particle, os.path.join(self.rollout_path, f'goal_particles'))

        if self.args.control_algo == 'fix':
            best_init_pose_seq, best_act_seq, best_model_loss = self.trajectory_optimization_with_horizon(self.args.n_grips, self.args.correction)
            best_sim_loss = self.visualize_results(best_init_pose_seq, best_act_seq, state_goal_final, self.args.n_grips)
        
        elif self.args.control_algo == 'search':
            model_loss_list = []
            sim_loss_list = []
            for grip_num in range(self.args.n_grips, 0, -1):
                init_pose_seq, act_seq, loss_seq = self.trajectory_optimization_with_horizon(
                    grip_num, self.args.correction)

                with open(f"{self.rollout_path}/init_pose_seq_{grip_num}.npy", 'wb') as f:
                    np.save(f, init_pose_seq)

                with open(f"{self.rollout_path}/act_seq_{grip_num}.npy", 'wb') as f:
                    np.save(f, act_seq)

                loss_sim = self.visualize_results(init_pose_seq, act_seq, state_goal_final, grip_num)
                model_loss_list.append([grip_num, loss_seq.item()])
                sim_loss_list.append([grip_num, loss_sim.item()])
                print(f"=============== With {grip_num} grips -> model_loss: {loss_seq}; sim_loss: {loss_sim} ===============")

                if grip_num == self.args.n_grips:
                    best_init_pose_seq = init_pose_seq
                    best_act_seq = act_seq
                    best_model_loss = loss_seq
                    best_sim_loss = loss_sim
                    best_idx = grip_num
                else:
                    if loss_sim < best_sim_loss:
                        best_init_pose_seq = init_pose_seq
                        best_act_seq = act_seq
                        best_model_loss = loss_seq
                        best_sim_loss = loss_sim
                        best_idx = grip_num

            visualize_rollout_loss([model_loss_list, sim_loss_list], os.path.join(self.rollout_path, f'rollout_loss'))
            os.system(f"cp {os.path.join(self.rollout_path, f'anim_{best_idx}.gif')} {os.path.join(self.rollout_path, f'best_anim.gif')}")
            os.system(f"cp {os.path.join(self.rollout_path, f'init_pose_seq_{best_idx}.npy')} {os.path.join(self.rollout_path, f'init_pose_seq_opt.npy')}")
            os.system(f"cp {os.path.join(self.rollout_path, f'act_seq_{best_idx}.npy')} {os.path.join(self.rollout_path, f'act_seq_opt.npy')}")

        elif self.args.control_algo == 'predict':
            checkpoint = None
            model_loss_list = []
            sim_loss_list = []
            n_iters = self.args.n_grips - self.args.predict_horizon + 1
            for i in range(n_iters):
                init_pose_seq, act_seq, loss_seq = self.trajectory_optimization_with_horizon(
                    self.args.predict_horizon, i==n_iters-1, checkpoint=checkpoint)
                checkpoint = [init_pose_seq[:1-self.args.predict_horizon], act_seq[:1-self.args.predict_horizon]]

                with open(f"{self.rollout_path}/init_pose_seq_{i}.npy", 'wb') as f:
                    np.save(f, init_pose_seq)

                with open(f"{self.rollout_path}/act_seq_{i}.npy", 'wb') as f:
                    np.save(f, act_seq)

                loss_sim = self.visualize_results(init_pose_seq, act_seq, state_goal_final, i)
                model_loss_list.append([i, loss_seq.item()])
                sim_loss_list.append([i, loss_sim.item()])
                print(f"=============== Iteration {i} -> model_loss: {loss_seq}; sim_loss: {loss_sim} ===============")
                
                if i == 0:
                    best_init_pose_seq = init_pose_seq
                    best_act_seq = act_seq
                    best_model_loss = loss_seq
                    best_sim_loss = loss_sim
                    best_idx = i
                else:
                    if loss_sim < best_sim_loss:
                        best_init_pose_seq = init_pose_seq
                        best_act_seq = act_seq
                        best_model_loss = loss_seq
                        best_sim_loss = loss_sim
                        best_idx = i

            visualize_rollout_loss([model_loss_list, sim_loss_list], os.path.join(self.rollout_path, f'rollout_loss'))
            os.system(f"cp {os.path.join(self.rollout_path, f'anim_{best_idx}.gif')} {os.path.join(self.rollout_path, f'best_anim.gif')}")
            os.system(f"cp {os.path.join(self.rollout_path, f'init_pose_seq_{best_idx}.npy')} {os.path.join(self.rollout_path, f'init_pose_seq_opt.npy')}")
            os.system(f"cp {os.path.join(self.rollout_path, f'act_seq_{best_idx}.npy')} {os.path.join(self.rollout_path, f'act_seq_opt.npy')}")

        return best_init_pose_seq.cpu(), best_act_seq.cpu(), best_model_loss.cpu(), best_sim_loss.cpu()


    def trajectory_optimization_with_horizon(self, grip_num, correction, checkpoint=None):
        if checkpoint:
            init_pose_seq, act_seq = checkpoint
        else:
            init_pose_seq = torch.Tensor()
            act_seq = torch.Tensor()

        for i in range(grip_num):
            self.grip_cur = f'{grip_num}-{i+1}'
            print(f'=============== {i+1}/{grip_num} ===============')

            if self.args.subgoal:
                state_goal = self.get_state_goal(i)
                n_grips_sample = 1
            else:
                state_goal = self.get_state_goal(self.args.n_grips - 1)
                n_grips_sample = grip_num - i

            init_pose_seqs_pool, act_seqs_pool = self.sample_action_params(n_grips_sample)
            
            if init_pose_seq.numel() == 0:
                init_pose_seq_cur = init_pose_seqs_pool[0]
                act_seq_cur = act_seqs_pool[0, 0, 0].unsqueeze(0).unsqueeze(0)
            else:
                init_pose_seq_cur = init_pose_seq
                act_seq_cur = act_seq

            # import pdb; pdb.set_trace()
            state_cur_sim = self.sim_rollout(init_pose_seq_cur.unsqueeze(0), act_seq_cur.unsqueeze(0))[0].squeeze()
            state_cur_sim_particles = state_cur_sim[:, :self.n_particle].clone()
            self.floor_state = state_cur_sim[:, self.n_particle: self.n_particle + task_params["n_shapes_floor"]].clone()

            visualize_points(state_cur_sim[-1], self.n_particle, os.path.join(self.rollout_path, f'sim_particles_{self.grip_cur}'))
            # visualize_points(state_cur_gt[-1], self.n_particle, os.path.join(self.rollout_path, f'gt_particles_{i}'))

            if checkpoint == None and i == 0:
                self.initial_state = state_cur_sim_particles

            if i == 0 or self.args.correction:
                state_cur = state_cur_sim_particles
            else:
                state_cur = state_seq_opt[-self.args.n_his:].clone()

            print(f"state_cur: {state_cur.shape}, state_goal: {state_goal.shape}")

            reward_seqs, model_state_seqs = self.rollout(init_pose_seqs_pool, act_seqs_pool, state_cur, state_goal)
            print('sampling: max: %.4f, mean: %.4f, std: %.4f' % (torch.max(reward_seqs), torch.mean(reward_seqs), torch.std(reward_seqs)))

            if self.args.opt_algo == 'max':
                init_pose_seq_opt, act_seq_opt, loss_opt, state_seq_opt = self.optimize_action_max(
                    init_pose_seqs_pool, act_seqs_pool, reward_seqs, model_state_seqs)
            
            elif self.args.opt_algo == 'CEM':
                for j in range(task_params["CEM_opt_iter"]):
                    self.CEM_opt_iter_cur = j
                    if j == task_params["CEM_opt_iter"] - 1:
                        init_pose_seq_opt, act_seq_opt, loss_opt, state_seq_opt = self.optimize_action_max(
                            init_pose_seqs_pool, act_seqs_pool, reward_seqs, model_state_seqs)
                    else:
                        init_pose_seqs_pool, act_seqs_pool = self.optimize_action_CEM(init_pose_seqs_pool, act_seqs_pool, reward_seqs)
                        reward_seqs, model_state_seqs = self.rollout(init_pose_seqs_pool, act_seqs_pool, state_cur, state_goal)
            
            elif self.args.opt_algo == "GD":
                with torch.set_grad_enabled(True):
                    init_pose_seq_opt, act_seq_opt, loss_opt, state_seq_opt = self.optimize_action_GD(init_pose_seqs_pool, act_seqs_pool, reward_seqs, state_cur, state_goal)
            
            elif self.args.opt_algo == "CEM_GD":
                for j in range(task_params["CEM_opt_iter"]):
                    self.CEM_opt_iter_cur = j
                    if j == task_params["CEM_opt_iter"] - 1:
                        with torch.set_grad_enabled(True):
                            init_pose_seq_opt, act_seq_opt, loss_opt, state_seq_opt = self.optimize_action_GD(init_pose_seqs_pool, act_seqs_pool, reward_seqs, state_cur, state_goal)
                    else:
                        init_pose_seqs_pool, act_seqs_pool = self.optimize_action_CEM(init_pose_seqs_pool, act_seqs_pool, reward_seqs)
                        reward_seqs, model_state_seqs = self.rollout(init_pose_seqs_pool, act_seqs_pool, state_cur, state_goal)

            else:
                raise NotImplementedError

            # pdb.set_trace()
            if not self.args.subgoal and correction:
                init_pose_seq_opt = init_pose_seq_opt[0].unsqueeze(0)
                act_seq_opt = act_seq_opt[0].unsqueeze(0)

            init_pose_seq = torch.cat((init_pose_seq, init_pose_seq_opt.clone()))
            act_seq = torch.cat((act_seq, act_seq_opt.clone()))
            loss_seq = loss_opt.clone()

            if not correction: 
                break

        return init_pose_seq, act_seq, loss_seq


    def visualize_results(self, init_pose_seq, act_seq, state_goal, i):
        model_state_seq = self.model_rollout(self.initial_state, init_pose_seq.unsqueeze(0), act_seq.unsqueeze(0))
        sample_state_seq, sim_state_seq = self.sim_rollout(init_pose_seq.unsqueeze(0), act_seq.unsqueeze(0))
        mode = '3d' if '3d' in self.args.data_type else '2d'
        model_state_seq = add_shapes(model_state_seq[0], init_pose_seq, act_seq, self.n_particle, mode=mode)
        sim_state_seq = add_shapes(sim_state_seq[0], init_pose_seq, act_seq, self.n_particle, mode=mode)

        sample_state_seq = sample_state_seq.squeeze()
        visualize_points(sample_state_seq[-1], self.n_particle, os.path.join(self.rollout_path, f'sim_particles_final_{i}'))
        plt_render([sim_state_seq, model_state_seq], state_goal[0], self.n_particle, os.path.join(self.rollout_path, f'anim_{i}.gif'))
        
        loss_sim = torch.neg(self.evaluate_traj(sample_state_seq[:, :self.n_particle].unsqueeze(0), state_goal, self.args.reward_type))

        emd_loss = torch.neg(self.evaluate_traj(sample_state_seq[:, :self.n_particle].unsqueeze(0), state_goal, 'emd'))
        chamfer_loss = torch.neg(self.evaluate_traj(sample_state_seq[:, :self.n_particle].unsqueeze(0), state_goal, 'chamfer'))
        print(f"EMD: {emd_loss}\nChamfer: {chamfer_loss}")

        return loss_sim


    def get_state_goal(self, i):
        goal_idx = min((i + 1) * (task_params["len_per_grip"] + task_params["len_per_grip_back"]) - 1, len(self.all_p) - 1)
        state_goal = torch.FloatTensor(self.all_p[goal_idx]).unsqueeze(0)[:, :self.n_particle, :]

        if len(self.args.goal_shape_name) > 0 and self.args.goal_shape_name != 'none' and self.args.goal_shape_name[:3] != 'vid':
            state_goal = self.goal_shapes[i]

        return state_goal


    def sample_action_params(self, n_grips):
        # np.random.seed(0)
        init_pose_seqs = []
        act_seqs = []
        n_sampled = 0
        while n_sampled < self.sample_size:
            init_pose_seq = []
            act_seq = []
            for i in range(n_grips):
                p_noise_x = task_params["p_noise_scale"] * (np.random.rand() * 2 - 1)
                p_noise_z = task_params["p_noise_scale"] * (np.random.rand() * 2 - 1)
                p_noise = np.clip(np.array([p_noise_x, 0, p_noise_z]), a_min=-0.1, a_max=0.1)
                new_mid_point = task_params["mid_point"][:3] + p_noise
                rot_noise = np.random.uniform(0, np.pi)
                z_angle = np.random.uniform(0, np.pi)
                print(new_mid_point, rot_noise, z_angle)
                mode = '3d' if '3d' in self.args.data_type else '2d'
                init_pose = get_pose(new_mid_point, rot_noise, z_angle, mode=mode)
                # print(init_pose.shape)
                init_pose_seq.append(init_pose)

                gripper_rate = np.random.uniform(*task_params["gripper_rate_limits"])
                actions = get_action_seq(rot_noise, gripper_rate)
                # print(actions.shape)
                act_seq.append(actions)

            init_pose_seq = torch.stack(init_pose_seq)
            init_pose_seqs.append(init_pose_seq)

            act_seq = torch.stack(act_seq)
            act_seqs.append(act_seq)

            n_sampled += 1

        return torch.stack(init_pose_seqs), torch.stack(act_seqs)


    def rollout(self, init_pose_seqs_pool, act_seqs_pool, state_cur, state_goal):
        # import pdb; pdb.set_trace()
        reward_seqs_rollout = []
        state_seqs_rollout = []
        
        n_batch = int(math.ceil(init_pose_seqs_pool.shape[0] / self.batch_size))
        batches = tqdm(range(n_batch), total=n_batch) if n_batch > 4 else range(n_batch)
        for i, _ in enumerate(batches):
            # print(f"Batch: {i}/{n_batch}")
            init_pose_seqs = init_pose_seqs_pool[i*self.batch_size:(i+1)*self.batch_size]
            act_seqs = act_seqs_pool[i*self.batch_size:(i+1)*self.batch_size]

            if self.args.use_sim:
                state_seqs, = self.sim_rollout(init_pose_seqs, act_seqs)
            else:
                state_seqs = self.model_rollout(state_cur, init_pose_seqs, act_seqs)
            
            reward_seqs = self.evaluate_traj(state_seqs, state_goal, self.args.reward_type)
            # print(f"reward seqs: {reward_seqs}")
            # reward_seqs = reward_seqs.data.cpu().numpy()

            reward_seqs_rollout.append(reward_seqs)
            state_seqs_rollout.append(state_seqs)

        # import pdb; pdb.set_trace()
        reward_seqs_rollout = torch.cat(reward_seqs_rollout, 0)
        state_seqs_rollout = torch.cat(state_seqs_rollout, 0)

        return reward_seqs_rollout, state_seqs_rollout


    def sim_rollout(self, init_pose_seqs, act_seqs):
        sample_state_seq_batch = []
        state_seq_batch = []
        for t in range(act_seqs.shape[0]):
            self.taichi_env.set_state(**self.env_init_state)
            state_seq = []
            for i in range(act_seqs.shape[1]):
                self.taichi_env.primitives.primitives[0].set_state(0, init_pose_seqs[t, i, task_params["gripper_mid_pt"], :7])
                self.taichi_env.primitives.primitives[1].set_state(0, init_pose_seqs[t, i, task_params["gripper_mid_pt"], 7:])
                for j in range(act_seqs.shape[2]):
                    self.taichi_env.step(act_seqs[t][i][j])
                    x = self.taichi_env.simulator.get_x(0)
                    step_size = len(x) // self.n_particle
                    # print(f"x before: {x.shape}")
                    x = x[::step_size]
                    particles = x[:self.n_particle]
                    # print(f"x after: {x.shape}")
                    state_seq.append(particles)

            sample_state = sample_particles(self.taichi_env, self.n_particle)
            state_seq_sample = []
            for i in range(self.args.n_his):
                state_seq_sample.append(sample_state)
            sample_state_seq_batch.append(np.stack(state_seq_sample))
            state_seq_batch.append(np.stack(state_seq))

        sample_state_seq_batch = torch.from_numpy(np.stack(sample_state_seq_batch))
        state_seq_batch = torch.from_numpy(np.stack(state_seq_batch))
        # print(f"sample_state_seq_batch: {sample_state_seq_batch.shape}")

        return sample_state_seq_batch, state_seq_batch


    def model_rollout(
        self,
        state_cur,      # [1, n_his, state_dim]
        init_pose_seqs,
        act_seqs,    # [n_sample, -1, action_dim]
    ):
        if not torch.is_tensor(init_pose_seqs):
            init_pose_seqs = torch.tensor(init_pose_seqs)

        if not torch.is_tensor(act_seqs):
            act_seqs = torch.tensor(act_seqs)

        init_pose_seqs = init_pose_seqs.float().to(device)
        act_seqs = act_seqs.float().to(device)
        state_cur = expand(init_pose_seqs.shape[0], state_cur.float().unsqueeze(0)).to(device)
        floor_state = expand(init_pose_seqs.shape[0], self.floor_state.float().unsqueeze(0)).to(device)

        memory_init = self.model.init_memory(init_pose_seqs.shape[0], self.n_particle + self.n_shape)
        scene_params = self.scene_params.expand(init_pose_seqs.shape[0], -1)
        group_gt = get_env_group(self.args, self.n_particle, scene_params, use_gpu=self.use_gpu)

        # pdb.set_trace()
        states_pred_list = []
        # act_seq n_sample, n_grip, grip_len, action_dim
        for i in range(act_seqs.shape[1]):
            # pdb.set_trace()
            shape1 = init_pose_seqs[:, i, :, :3]
            shape2 = init_pose_seqs[:, i, :, 7:10]
            state_cur = torch.cat([state_cur[:, :, :self.n_particle, :], floor_state, shape1.unsqueeze(1).expand([-1, self.args.n_his, -1, -1]), 
                        shape2.unsqueeze(1).expand([-1, self.args.n_his, -1, -1])], dim=2)
            for j in range(act_seqs.shape[2]):
                attrs = []
                Rr_curs = []
                Rs_curs = []
                Rn_curs = []
                max_n_rel = 0
                for k in range(act_seqs.shape[0]):
                    # pdb.set_trace()
                    state_last = state_cur[k][-1]
                    attr, _, Rr_cur, Rs_cur, Rn_cur, cluster_onehot = prepare_input(state_last.detach().cpu().numpy(), self.n_particle,
                                                                self.n_shape, self.args, stdreg=self.args.stdreg)
                    attr = attr.to(device)
                    Rr_cur = Rr_cur.to(device)
                    Rs_cur = Rs_cur.to(device)
                    Rn_cur = Rn_cur.to(device)
                    max_n_rel = max(max_n_rel, Rr_cur.size(0))
                    attr = attr.unsqueeze(0)
                    Rr_cur = Rr_cur.unsqueeze(0)
                    Rs_cur = Rs_cur.unsqueeze(0)
                    Rn_cur = Rn_cur.unsqueeze(0)
                    attrs.append(attr)
                    Rr_curs.append(Rr_cur)
                    Rs_curs.append(Rs_cur)
                    Rn_curs.append(Rn_cur)

                attrs = torch.cat(attrs, dim=0)
                for k in range(len(Rr_curs)):
                    Rr, Rs, Rn = Rr_curs[k], Rs_curs[k], Rn_curs[k]
                    Rr = torch.cat([Rr, torch.zeros((1, max_n_rel - Rr.size(1), self.n_particle + self.n_shape)).to(device)], 1)
                    Rs = torch.cat([Rs, torch.zeros((1, max_n_rel - Rs.size(1), self.n_particle + self.n_shape)).to(device)], 1)
                    Rn = torch.cat([Rn, torch.zeros((1, max_n_rel - Rn.size(1), self.n_particle + self.n_shape)).to(device)], 1)
                    Rr_curs[k], Rs_curs[k], Rn_curs[k] = Rr, Rs, Rn

                Rr_curs = torch.cat(Rr_curs, dim=0)
                Rs_curs = torch.cat(Rs_curs, dim=0)
                Rn_curs = torch.cat(Rn_curs, dim=0)

                inputs = [attrs, state_cur, Rr_curs, Rs_curs, Rn_curs, memory_init, group_gt, None]

                # pdb.set_trace()
                pred_pos, pred_motion_norm, std_cluster = self.model.predict_dynamics(inputs)

                shape1 += act_seqs[:, i, j, :3].unsqueeze(1).expand(-1, task_params["n_shapes_per_gripper"], -1) * 0.02
                shape2 += act_seqs[:, i, j, 6:9].unsqueeze(1).expand(-1, task_params["n_shapes_per_gripper"], -1) * 0.02

                pred_pos = torch.cat([pred_pos, state_cur[:, -1, self.n_particle: self.n_particle + task_params["n_shapes_floor"], :], shape1, shape2], 1)
                # print(f"pred_pos shape: {pred_pos.shape}")

                state_cur = torch.cat([state_cur[:, 1:], pred_pos.unsqueeze(1)], 1)
                # print(f"state_cur shape: {state_cur.shape}")
                # print(torch.cuda.memory_summary())
                
                states_pred_list.append(pred_pos[:, :self.n_particle, :])
        
        states_pred_array = torch.stack(states_pred_list, dim=1).cpu()

        # print(f"torch mem allocated: {torch.cuda.memory_allocated()}; torch mem reserved: {torch.cuda.memory_reserved()}")

        return states_pred_array


    def evaluate_traj(
        self,
        state_seqs,     # [n_sample, n_look_ahead, state_dim]
        state_goal,     # [state_dim]
        reward_type
    ):
        # print(state_seqs.shape, state_goal.shape)
        reward_seqs = []
        for i in range(state_seqs.shape[0]):
            state_final = state_seqs[i, -1].unsqueeze(0)
            if state_final.shape != state_goal.shape:
                print("Data shape doesn't match in evaluate_traj!")
                raise ValueError

            # smaller loss, larger reward
            if reward_type == "emd":
                loss = emd_loss(state_final, state_goal)
            elif reward_type == "chamfer":
                loss = chamfer_loss(state_final, state_goal)
            elif reward_type == "emd_chamfer_h":
                emd_weight, chamfer_weight, h_weight = task_params["loss_weights"]
                loss = 0
                if emd_weight > 0:
                    loss += emd_weight * emd_loss(state_final, state_goal)
                if chamfer_weight > 0:
                    loss += chamfer_weight * chamfer_loss(state_final, state_goal)
                if h_weight > 0:
                    loss += h_weight * h_loss(state_final, state_goal)
            else:
                raise NotImplementedError

            reward_seqs.append(0.0 - loss)

        reward_seqs = torch.stack(reward_seqs)
        return reward_seqs


    def optimize_action_max(
        self,
        init_pose_seqs,
        act_seqs,       # [n_sample, -1, action_dim]
        reward_seqs,    # [n_sample]
        state_seqs
    ):

        idx = torch.argsort(reward_seqs)
        loss_opt = torch.neg(reward_seqs[idx[-1]]).view(1)
        print(f"Selected idx: {idx[-1]} with reward {reward_seqs[idx[-1]]}")

        visualize_sampled_init_pos(init_pose_seqs, reward_seqs, idx, \
            os.path.join(self.rollout_path, f'plot_max_{self.grip_cur}'))

        # pdb.set_trace()
        return init_pose_seqs[idx[-1]], act_seqs[idx[-1]], loss_opt, state_seqs[idx[-1]]


    def optimize_action_CEM(    # Cross Entropy Method (CEM)
        self,
        init_pose_seqs,
        act_seqs,
        reward_seqs,    # [n_sample]
        best_k_ratio=0.1
    ):
        best_k = max(4, int(init_pose_seqs.shape[0] * best_k_ratio))
        idx = torch.argsort(reward_seqs)
        print(f"Selected top reward seqs: {reward_seqs[idx[-best_k:]]}")
        # print(f"Selected top init pose seqs: {init_pose_seqs[idx[-best_k:], :, task_params["gripper_mid_pt"], :7]}")

        visualize_sampled_init_pos(init_pose_seqs, reward_seqs, idx, \
            os.path.join(self.rollout_path, f'plot_cem_s{self.grip_cur}_o{self.CEM_opt_iter_cur}'))

        # pdb.set_trace()
        init_pose_seqs_pool = []
        act_seqs_pool = []
        for i in range(best_k, 0, -1):
            init_pose_seq = init_pose_seqs[idx[-i]]
            mid_point_seq, angle_seq, z_angle_seq = get_params_from_pose(init_pose_seq)
            act_seq = act_seqs[idx[-i]]
            # gripper_rate_seq = get_gripper_rate_from_action_seq(act_seq)
            # print(f"Selected init pose seq: {init_pose_seq[:, task_params["gripper_mid_pt"], :7]}")
            init_pose_seqs_pool.append(init_pose_seq)
            act_seqs_pool.append(act_seq)

            for k in range(task_params["CEM_gripper_rate_sample_size"] - 1):
                # gripper_rate_noise = torch.clamp(torch.tensor(np.random.randn(init_pose_seq.shape[0])*0.02), max=0.05, min=-0.05)
                # gripper_rate_sample = gripper_rate_seq + gripper_rate_noise
                gripper_rate_sample = []
                for s in range(init_pose_seq.shape[0]):
                    gripper_rate_sample.append(np.random.uniform(*task_params["gripper_rate_limits"]))
                gripper_rate_sample = torch.tensor(gripper_rate_sample)
                # print(f"{i} gripper_rate_sample: {gripper_rate_sample}")
                act_seq_sample = get_action_seq_from_pose(init_pose_seq, gripper_rate_sample)

                init_pose_seqs_pool.append(init_pose_seq)
                act_seqs_pool.append(act_seq_sample)

            if i > 1:
                n_init_pose_samples = task_params["CEM_init_pose_sample_size"] // (2**i)
            else:
                n_init_pose_samples = task_params["CEM_init_pose_sample_size"] - len(init_pose_seqs_pool) // (task_params["CEM_gripper_rate_sample_size"]) + 1
            
            j = 1
            while j < n_init_pose_samples:
                init_pose_seq_sample = []
                for k in range(init_pose_seq.shape[0]):
                    p_noise = torch.clamp(torch.tensor([0, 0, np.random.randn()*0.03]), min=-0.1, max=0.1)
                    rot_noise = torch.clamp(torch.tensor(np.random.randn()*math.pi/36), min=-0.1, max=0.1)
                    z_angle_noise =  torch.clamp(torch.tensor(np.random.randn()*math.pi/36), min=-0.1, max=0.1)

                    new_mid_point = mid_point_seq[k, :3] + p_noise
                    new_angle = angle_seq[k] + rot_noise
                    new_z_angle = z_angle_seq[k] + z_angle_noise
                    mode = '3d' if '3d' in self.args.data_type else '2d'
                    init_pose = get_pose(new_mid_point, new_angle, new_z_angle, mode=mode)
                    init_pose_seq_sample.append(init_pose)

                init_pose_seq_sample = torch.stack(init_pose_seq_sample)
                # print(f"Selected init pose seq: {init_pose_seq_sample[:, task_params["gripper_mid_pt"], :7]}")

                # pdb.set_trace()
                for k in range(task_params["CEM_gripper_rate_sample_size"]):
                    # gripper_rate_noise = torch.tensor(np.random.randn(init_pose_seq.shape[0])*0.01).to(device)
                    # gripper_rate_sample = torch.clamp(gripper_rate_seq + gripper_rate_noise, max=task_params["gripper_rate_limits"][1], min=task_params["gripper_rate_limits"][0])
                    gripper_rate_sample = []
                    for s in range(init_pose_seq.shape[0]):
                        gripper_rate_sample.append(np.random.uniform(*task_params["gripper_rate_limits"]))
                    gripper_rate_sample = torch.tensor(gripper_rate_sample)
                    # print(f"gripper_rate_sample: {gripper_rate_sample}")
                    act_seq_sample = get_action_seq_from_pose(init_pose_seq_sample, gripper_rate_sample)

                    init_pose_seqs_pool.append(init_pose_seq_sample)
                    act_seqs_pool.append(act_seq_sample)

                j += 1

        # pdb.set_trace()
        init_pose_seqs_pool = torch.stack(init_pose_seqs_pool)
        act_seqs_pool = torch.stack(act_seqs_pool)
        print(f"Init pose seq pool shape: {init_pose_seqs_pool.shape}; Act seq pool shape: {act_seqs_pool.shape}")

        return init_pose_seqs_pool, act_seqs_pool


    def optimize_action_GD(
        self,
        init_pose_seqs,
        act_seqs,
        reward_seqs,
        state_cur,
        state_goal,
        lr=1e-1,
        best_k_ratio=0.1
    ):
        idx = torch.argsort(reward_seqs)
        best_k = max(4, int(reward_seqs.shape[0] * best_k_ratio))
        print(f"Selected top reward seqs: {reward_seqs[idx[-best_k:]]}")

        best_mid_point_seqs = []
        best_angle_seqs = [] 
        best_z_angle_seqs = []
        best_gripper_rate_seqs = []
        for i in range(best_k, 0, -1):
            best_mid_point_seq, best_angle_seq, best_z_angle_seq = get_params_from_pose(init_pose_seqs[idx[-i]])
            best_gripper_rate_seq = get_gripper_rate_from_action_seq(act_seqs[idx[-i]])

            best_mid_point_seqs.append(best_mid_point_seq)
            best_angle_seqs.append(best_angle_seq)
            best_z_angle_seqs.append(best_z_angle_seq)
            best_gripper_rate_seqs.append(best_gripper_rate_seq)
        
        # pdb.set_trace()
        best_mid_point_seqs = torch.stack(best_mid_point_seqs)
        best_angle_seqs = torch.stack(best_angle_seqs)
        best_z_angle_seqs = torch.stack(best_z_angle_seqs)
        best_gripper_rate_seqs = torch.stack(best_gripper_rate_seqs)

        mid_points = best_mid_point_seqs.requires_grad_()
        angles = best_angle_seqs.requires_grad_()
        z_angles = best_z_angle_seqs.requires_grad_()
        gripper_rates = best_gripper_rate_seqs.requires_grad_()

        mid_point_x_bounds = [task_params["mid_point"][0] - task_params["p_noise_bound"], task_params["mid_point"][0] + task_params["p_noise_bound"]]
        mid_point_z_bounds = [task_params["mid_point"][2] - task_params["p_noise_bound"], task_params["mid_point"][2] + task_params["p_noise_bound"]]

        loss_list_all = []
        n_batch = int(math.ceil(best_k / task_params["GD_batch_size"]))
        reward_list = None
        model_state_seq_list = None

        for b in range(n_batch):
            print(f"Batch {b}/{n_batch}:")

            optimizer = torch.optim.LBFGS([mid_points, angles, gripper_rates], lr=lr, tolerance_change=1e-5, line_search_fn="strong_wolfe")
            
            start_idx = b * task_params["GD_batch_size"]
            end_idx = (b + 1) * task_params["GD_batch_size"]

            epoch = 0
            loss_list = []
            reward_seqs = None
            model_state_seqs = None
            
            def closure():
                nonlocal epoch
                nonlocal loss_list
                nonlocal reward_seqs
                nonlocal model_state_seqs

                # print(f"Params:\nmid_point: {mid_points}\nangle: {angles}\ngripper_rate: {gripper_rates}")
                
                optimizer.zero_grad()

                init_pose_seq_samples = []
                act_seq_samples = []
                for i in range(mid_points[start_idx:end_idx].shape[0]):
                    init_pose_seq_sample = []
                    for j in range(mid_points.shape[1]):
                        # pdb.set_trace()
                        mid_point_clipped_x = torch.clamp(mid_points[start_idx + i, j, 0], min=mid_point_x_bounds[0], max=mid_point_x_bounds[1])
                        mid_point_clipped_z = torch.clamp(mid_points[start_idx + i, j, 2], min=mid_point_z_bounds[0], max=mid_point_z_bounds[1])
                        mid_point_clipped = [mid_point_clipped_x, mid_points[start_idx + i, j, 1], mid_point_clipped_z]
                        mode = '3d' if '3d' in self.args.data_type else '2d'
                        init_pose = get_pose(mid_point_clipped, angles[start_idx + i, j], z_angles[start_idx + i, j], mode=mode)
                        init_pose_seq_sample.append(init_pose)

                    # pdb.set_trace()
                    init_pose_seq_sample = torch.stack(init_pose_seq_sample)

                    gripper_rate_clipped = torch.clamp(gripper_rates[start_idx + i], min=0, max=task_params["gripper_rate_limits"][1])
                    act_seq_sample = get_action_seq_from_pose(init_pose_seq_sample, gripper_rate_clipped)

                    init_pose_seq_samples.append(init_pose_seq_sample)
                    act_seq_samples.append(act_seq_sample)

                init_pose_seq_samples = torch.stack(init_pose_seq_samples)
                act_seq_samples = torch.stack(act_seq_samples)

                # pdb.set_trace()
                reward_seqs, model_state_seqs = self.rollout(init_pose_seq_samples, act_seq_samples[:, :, :task_params["len_per_grip"], :], state_cur, state_goal)

                loss = torch.sum(torch.neg(reward_seqs))
                loss_list.append([epoch, loss.item()])
                
                loss.backward()
                # gripper_rates.grad[start_idx + i] /= task_params["len_per_grip"]

                epoch += 1

                return loss

            loss = optimizer.step(closure)

            loss_list_all.append(loss_list)

            print(f"reward seqs after {len(loss_list)} iterations: {reward_seqs}")
            reward_list = torch.cat((reward_list, reward_seqs)) if reward_list != None else reward_seqs
            model_state_seq_list = torch.cat((model_state_seq_list, model_state_seqs)) if model_state_seq_list != None else model_state_seqs

        loss_min = torch.min(torch.neg(reward_list))
        print(f"Loss: {loss_min}")

        # print(f"Params:\nmid_point: {mid_points}\nangle: {angles}\ngripper_rate: {gripper_rates}")

        visualize_loss(loss_list_all, os.path.join(self.rollout_path, f'plot_GD_loss_{self.grip_cur}'))

        # pdb.set_trace()
        idx = torch.argsort(reward_list)
        loss_opt = torch.neg(reward_list[idx[-1]]).view(1)
        mid_point_clipped_opt = []
        init_pose_seq_opt = []
        for i in range(mid_points[idx[-1]].shape[0]):
            mid_point_clipped_x = torch.clamp(mid_points[idx[-1], i, 0], min=mid_point_x_bounds[0], max=mid_point_x_bounds[1])
            mid_point_clipped_z = torch.clamp(mid_points[idx[-1], i, 2], min=mid_point_z_bounds[0], max=mid_point_z_bounds[1])
            mid_point_clipped = [mid_point_clipped_x, mid_points[idx[-1], i, 1], mid_point_clipped_z]
            mid_point_clipped_opt.append(mid_point_clipped)
            mode = '3d' if '3d' in self.args.data_type else '2d'
            init_pose = get_pose(mid_point_clipped, angles[idx[-1], i], z_angles[idx[-1], i], mode=mode)
            init_pose_seq_opt.append(init_pose)

        init_pose_seq_opt = torch.stack(init_pose_seq_opt)

        gripper_rate_opt = torch.clamp(gripper_rates[idx[-1]], min=0, max=task_params["gripper_rate_limits"][1])
        act_seq_opt = get_action_seq_from_pose(init_pose_seq_opt, gripper_rate_opt)

        print(f"Optimal set of params:\nmid_point: {torch.tensor(mid_point_clipped_opt)}")
        print(f"angle: {angles[idx[-1]]}\nz_angle: {z_angles[idx[-1]]}\ngripper_rate: {gripper_rate_opt}")
        print(f"Optimal init pose seq: {init_pose_seq_opt[:, task_params['gripper_mid_pt'], :7]}")

        return init_pose_seq_opt, act_seq_opt, loss_opt, model_state_seq_list[idx[-1]]


init_pose_gt = []
act_seq_gt = []

def main():
    global init_pose_gt
    global act_seq_gt

    args = gen_args()
    set_seed(args.random_seed)
    args.outf = os.path.dirname(args.model_path)

    # Update task params
    if not 'fixed' in args.data_type:
        task_params["p_noise_scale"] = 0.03

    if 'small' in args.data_type:
        task_params['tool_size'] = 0.03

    task_params["gripper_rate_limits"] = [
        (task_params['sample_radius'] * 2 - (task_params['gripper_gap_limits'][0] + 2 * task_params['tool_size'])) / (2 * task_params['len_per_grip']),
        (task_params['sample_radius'] * 2 - (task_params['gripper_gap_limits'][1] + 2 * task_params['tool_size'])) / (2 * task_params['len_per_grip'])
    ]

    if len(args.controlf) > 0:
        args.outf = args.controlf

    if args.gt_action:
        test_name = f'sim_{args.use_sim}+gt_action_{args.gt_action}+{args.reward_type}'
    else:
        test_name = f'sim_{args.use_sim}+{args.shape_type}+algo_{args.control_algo}+{args.n_grips}_grips+{args.opt_algo}+{args.reward_type}+correction_{args.correction}+debug_{args.debug}'

    if len(args.goal_shape_name) > 0 and args.goal_shape_name != 'none':
        vid_idx = 0
        if args.goal_shape_name[:3] == 'vid':
            vid_idx = int(args.goal_shape_name[4:])
            shape_goal_dir = str(vid_idx).zfill(3)
        else:
            shape_goal_dir = args.goal_shape_name
    else:
        print("Please specify a valid goal shape name!")
        raise ValueError

    control_out_dir = os.path.join(args.outf, 'control', shape_goal_dir, test_name)
    os.system('rm -r ' + control_out_dir)
    os.system('mkdir -p ' + control_out_dir)

    tee = Tee(os.path.join(control_out_dir, 'control.log'), 'w')

    # set up the env
    cfg = load(args.gripperf)
    print(cfg)

    env = None
    state = None
    if platform != 'darwin':
        env = TaichiEnv(cfg, nn=False, loss=False)
        env.initialize()
        state = env.get_state()

        env.set_state(**state)
        taichi_env = env

        env.renderer.camera_pos[0] = 0.5
        env.renderer.camera_pos[1] = 2.5
        env.renderer.camera_pos[2] = 0.5
        env.renderer.camera_rot = (1.57, 0.0)

        env.primitives.primitives[0].set_state(0, [0.3, 0.4, 0.5, 1, 0, 0, 0])
        env.primitives.primitives[1].set_state(0, [0.7, 0.4, 0.5, 1, 0, 0, 0])
        
        def set_parameters(env: TaichiEnv, yield_stress, E, nu):
            env.simulator.yield_stress.fill(yield_stress)
            _mu, _lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
            env.simulator.mu.fill(_mu)
            env.simulator.lam.fill(_lam)

        set_parameters(env, yield_stress=200, E=5e3, nu=0.2) # 200， 5e3, 0.2

        def update_camera(env):
            env.renderer.camera_pos[0] = 0.5 #np.array([float(i) for i in (0.5, 2.5, 0.5)]) #(0.5, 2.5, 0.5)  #.from_numpy(np.array([[0.5, 2.5, 0.5]]))
            env.renderer.camera_pos[1] = 2.5
            env.renderer.camera_pos[2] = 2.2
            env.renderer.camera_rot = (0.8, 0.0)
            env.render_cfg.defrost()
            env.render_cfg.camera_pos_1 = (0.5, 2.5, 2.2)
            env.render_cfg.camera_rot_1 = (0.8, 0.)
            env.render_cfg.camera_pos_2 = (2.4, 2.5, 0.2)
            env.render_cfg.camera_rot_2 = (0.8, 1.8)
            env.render_cfg.camera_pos_3 = (-1.9, 2.5, 0.2)
            env.render_cfg.camera_rot_3 = (0.8, -1.8)
            env.render_cfg.camera_pos_4 = (0.5, 2.5, -1.8)
            env.render_cfg.camera_rot_4 = (0.8, 3.14)

        update_camera(env)

        # update the tool size
        env.primitives.primitives[0].r[None] = task_params['tool_size']
        env.primitives.primitives[1].r[None] = task_params['tool_size']


    # load dynamics model
    model = Model(args, use_gpu)
    print("model_kp #params: %d" % count_parameters(model))
    if use_gpu:
        pretrained_dict = torch.load(args.model_path)
    else:
        pretrained_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    # only load parameters in dynamics_predictor
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() \
        if 'dynamics_predictor' in k and k in model_dict}
    model.load_state_dict(pretrained_dict, strict=False)
    model.eval()
    
    model = model.to(device)


    # load data (state, actions)
    unit_quat_pad = np.tile([1, 0, 0, 0], (task_params["n_shapes_per_gripper"], 1))
    task_name = 'ngrip_fixed'
    data_names = ['positions', 'shape_quats', 'scene_params']
    rollout_dir = f"./data/data_{args.data_type}/train/"
    steps_per_grip = task_params["len_per_grip"] + task_params["len_per_grip_back"]

    # init_pose_gt = []
    # act_seq_gt = []
    all_p = []
    all_s = []
    actions = []
    frame_list = sorted(glob.glob(os.path.join(args.dataf, 'train', str(vid_idx).zfill(3), 'shape_*.h5')))
    gt_frame_list = sorted(glob.glob(os.path.join(args.dataf, 'train', str(vid_idx).zfill(3), 'shape_gt_*.h5')))
    args.time_step = (len(frame_list) - len(gt_frame_list))
    for t in range(args.time_step):
        frame_name = str(t) + '.h5'
        if args.gt_state_goal:
            frame_name = 'gt_' + frame_name
        if args.shape_aug:
            frame_name = 'shape_' + frame_name
        frame_path = os.path.join(rollout_dir, str(vid_idx).zfill(3), frame_name) 
        this_data = load_data(data_names, frame_path)

        n_particle, n_shape, scene_params = get_scene_info(this_data)
        scene_params = torch.FloatTensor(scene_params).unsqueeze(0)
        g1_idx = n_particle + task_params["n_shapes_floor"]
        g2_idx = g1_idx + task_params["n_shapes_per_gripper"]

        states = this_data[0]
        
        if t >= 1:
            all_p.append(states)
            all_s.append(this_data[1])

            action = np.concatenate([(states[g1_idx] - prev_states[g1_idx]) / 0.02, np.zeros(3),
                                    (states[g2_idx] - prev_states[g2_idx]) / 0.02, np.zeros(3)])
        
            if len(actions) == task_params["len_per_grip"] - 1:
                actions.insert(0, actions[0])
            elif len(actions) == steps_per_grip - 1:
                actions.append(actions[-1])
            else:
                actions.append(action)

            if t == 1: actions.insert(0, actions[0])
            if t == args.time_step - 1: actions.append(actions[-1])

        prev_states = states

        if t % steps_per_grip == 0:
            init_pose_gt.append(np.concatenate((states[g1_idx: g2_idx], unit_quat_pad, states[g2_idx:], unit_quat_pad), axis=1))
        
        if len(actions) == steps_per_grip:
            # print(f"Actions: {actions}")
            act_seq_gt.append(actions)
            # import pdb; pdb.set_trace()
            # hard code
            init_pose_gt[-1] = np.concatenate((init_pose_gt[-1][:, :3] - 2 * 0.02 * np.tile(actions[0][:3], (init_pose_gt[-1].shape[0], 1)), unit_quat_pad, \
                init_pose_gt[-1][:, 7:10] - 2 * 0.02 * np.tile(actions[0][6:9], (init_pose_gt[-1].shape[0], 1)), unit_quat_pad), axis=1)
            actions = []

        prev_states = states

    init_pose_gt = np.expand_dims(init_pose_gt, axis=0)
    act_seq_gt = np.expand_dims(act_seq_gt, axis=0)

    print(f"GT shape: init pose: {init_pose_gt.shape}; actions: {act_seq_gt.shape}")
    print(f"GT init pose: {init_pose_gt[0, :, task_params['gripper_mid_pt'], :7]}")
    # print(act_seq_gt)

    # load goal shape
    if len(args.goal_shape_name) > 0 and args.goal_shape_name != 'none' and args.goal_shape_name[:3] != 'vid':
        # if len(args.goal_shape_name) > 1:
        #     shape_type = 'simple'
        # else:
        #     shape_type = "alphabet"
        shape_dir = os.path.join(os.getcwd(), 'shapes', args.shape_type, args.goal_shape_name)
        goal_shapes = []
        for i in range(args.n_grips):
            if args.subgoal:
                goal_frame_name = f'{args.goal_shape_name}_{i}.h5'
            else:
                goal_frame_name = f'{args.goal_shape_name}.h5'
            # if args.shape_aug:
            #     goal_frame_name = 'shape_' + goal_frame_name
            goal_frame_path = os.path.join(shape_dir, goal_frame_name)
            goal_data = load_data(data_names, goal_frame_path)
            goal_shapes.append(torch.FloatTensor(goal_data[0]).unsqueeze(0)[:, :n_particle, :])
    else:
        goal_shapes = torch.FloatTensor(all_p[-1]).unsqueeze(0)[:, :n_particle, :]

    # trajecotory optimization
    planner = Planner(args=args, taichi_env=env, env_init_state=state, scene_params=scene_params, n_particle=n_particle, 
                    n_shape=n_shape, model=model, all_p=all_p, goal_shapes=goal_shapes, task_params=task_params, use_gpu=use_gpu, 
                    rollout_path=control_out_dir)

    with torch.no_grad():
        if args.gt_action:
            state_cur = torch.FloatTensor(np.stack(all_p[:args.n_his]))
            state_goal = torch.FloatTensor(all_p[-1]).unsqueeze(0)[:, :n_particle, :]
            if args.use_sim:
                state_seqs, = planner.sim_rollout(init_pose_gt, act_seq_gt)
            else:
                init_pose_gt_batch = np.repeat(init_pose_gt, args.shooting_batch_size, axis=0)
                act_seq_gt_batch = np.repeat(act_seq_gt, args.shooting_batch_size, axis=0)
                state_seqs = planner.model_rollout(state_cur, init_pose_gt_batch, act_seq_gt_batch)
            reward_seqs = planner.evaluate_traj(state_seqs, state_goal, args.reward_type)
            print(f"GT reward: {reward_seqs}")
            init_pose_seq = init_pose_gt[0]
            act_seq = act_seq_gt[0]
        else:
            init_pose_seq, act_seq, loss_seq, loss_sim_seq = planner.trajectory_optimization()

    print(init_pose_seq.shape, act_seq.shape)
    print(f"Best init pose: {init_pose_seq[:, task_params['gripper_mid_pt'], :7]}")
    print(f"Best model loss: {loss_seq}; Best sim loss: {loss_sim_seq}")

    # with open(f"{control_out_dir}/init_pose_seq_opt.npy", 'wb') as f:
    #     np.save(f, init_pose_seq)

    # with open(f"{control_out_dir}/act_seq_opt.npy", 'wb') as f:
    #     np.save(f, act_seq)


if __name__ == '__main__':
    main()
