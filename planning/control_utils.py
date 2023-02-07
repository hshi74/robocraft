import numpy as np
import torch

from datetime import datetime
from utils.data_utils import *
from pytorch3d.transforms import *
from transforms3d.quaternions import *


def normalize_state(args, state_cur, state_goal, pkg='numpy'):
    if len(state_cur.shape) == 2:
        dim = 0 
    elif len(state_cur.shape) == 3:
        dim = 1
    else:
        raise NotImplementedError

    if pkg == 'numpy':
        mean_p = np.mean(state_cur, axis=dim)
        std_p = np.std(state_cur, axis=dim)
        state_cur_norm = state_cur - mean_p
        
        mean_goal = np.mean(state_goal, axis=dim)
        std_goal = np.std(state_goal, axis=dim)
        # if 'alphabet' in args.target_shape_name:    
        #     # state_goal_norm = (state_goal - mean_goal) / std_goal
        #     if dim == 0:
        #         std_goal_new = np.concatenate((std_p[:2], std_goal[2:]))
        #     else:
        #         std_goal_new = np.concatenate((std_p[:, :2], std_goal[:, 2:]), axis=1)
        
        #     state_goal_norm = (state_goal - mean_goal) / std_goal_new
        # else:
        state_goal_norm = state_goal - mean_goal
    else:
        mean_p = torch.mean(state_cur, dim=dim)
        std_p = torch.std(state_cur, dim=dim)
        state_cur_norm = state_cur - mean_p
        
        mean_goal = torch.mean(state_goal, dim=dim)
        std_goal = torch.std(state_goal, dim=dim)
        # if 'alphabet' in args.target_shape_name:    
        #     # state_goal_norm = (state_goal - mean_goal) / std_goal
        #     if dim == 0:
        #         std_goal_new = torch.cat((std_p[:2], std_goal[2:]))
        #     else:
        #         std_goal_new = torch.cat((std_p[:, :2], std_goal[:, 2:]), dim=1)
        
        #     state_goal_norm = (state_goal - mean_goal) / std_goal_new
        # else:
        state_goal_norm = state_goal - mean_goal

    return state_cur_norm, state_goal_norm


def get_param_bounds(tool_params, min_bounds, max_bounds):
    param_bounds = []
    r = min(max_bounds[:2] - min_bounds[:2]) / 2
    param_bounds.append([-0.01, 0.04]) # r
    param_bounds.append(tool_params["theta_range"])
    param_bounds.append(tool_params["phi1_range"])    
    param_bounds.append(tool_params["phi2_range"])
    param_bounds.append([tool_params["grip_min"], min(2*r, 0.08)])

    return torch.FloatTensor(np.array(param_bounds))


def param_seqs_to_init_poses(args, center, plan_params, param_seqs, is_3d=False):
    device = param_seqs.device
    dough_center = center.to(device)
    B = param_seqs.shape[0] * param_seqs.shape[1]
    ps = param_seqs.view(B, -1)
    ee_fingertip_T_mat = expand(B, torch.tensor(args.ee_fingertip_T_mat, dtype=torch.float32, device=device).unsqueeze(0))
    
    a = torch.tensor([0, 1, 0, 0], dtype=torch.float32, device=device)
    rmat = expand(B, quaternion_to_matrix(a).unsqueeze(0))
    
    b = expand(B, torch.tensor([[[0, 0, 0, 1]]], dtype=torch.float32, device=device))

    if is_3d:        
        r_z = torch.tile(torch.tensor([0, 0, 0 - np.pi / 4], dtype=torch.float32, device=device), (B, 1))
        rmat_z = axis_angle_to_matrix(r_z)

        r_x_scaled = ps[:, 3] / np.sqrt(2)
        r_x = torch.stack((r_x_scaled, r_x_scaled, torch.zeros_like(r_x_scaled)), dim=-1)
        rmat_x = axis_angle_to_matrix(r_x)

        ee_rot = rmat.bmm(rmat_z.bmm(rmat_x))

        pos_delta = torch.stack(
            ((-ps[:, 0] * torch.sin(ps[:, 2]) + args.tool_center_z * torch.sin(ps[:, 3])) * 0.0, # torch.cos(0 - np.pi / 2) 
            (-ps[:, 0] * torch.sin(ps[:, 2]) + args.tool_center_z * torch.sin(ps[:, 3])) * -1.0, # torch.sin(0 - np.pi / 2)
            ps[:, 0] * torch.cos(ps[:, 2]) + args.tool_center_z * torch.cos(ps[:, 3])), dim=-1
        )
    else:
        r_z = torch.stack((torch.zeros_like(ps[:, 1]), torch.zeros_like(ps[:, 1]), ps[:, 1] - np.pi / 4), dim=-1)
        rmat_z = axis_angle_to_matrix(r_z)

        ee_rot = rmat.bmm(rmat_z)

        pos_delta = torch.stack(
            (torch.mul(ps[:, 0] * torch.sin(ps[:, 2]) + args.tool_center_z * 0.0, torch.sin(ps[:, 1])),
            torch.mul(ps[:, 0] * torch.sin(ps[:, 2]) + args.tool_center_z * 0.0, torch.cos(ps[:, 1])),
            ps[:, 0] * torch.cos(ps[:, 2]) + args.tool_center_z * 1.0), dim=-1
        )

    fingermid_pos = pos_delta + dough_center
    fingertip_mat = ee_rot.bmm(ee_fingertip_T_mat[:, :3, :3])

    # import pdb; pdb.set_trace()
    fingertip_T_list = []
    for k in range(len(args.tool_dim[args.env])):
        offset = torch.tensor([0, (2 * k - 1) * (plan_params["init_grip"] / 2), 0], dtype=torch.float32, device=device)
        offset_batch = expand(B, offset.unsqueeze(0)).unsqueeze(-1)
        fingertip_pos = fingertip_mat.bmm(offset_batch).squeeze() + fingermid_pos

        fingertip_T = torch.cat((torch.cat((fingertip_mat, fingertip_pos.unsqueeze(-1)), dim=-1), b), dim=1)
        fingertip_T_list.append(fingertip_T)

    fingertip_T_batch = torch.swapaxes(torch.stack(fingertip_T_list), 0, 1)
    
    init_pose_seq = []
    for i in range(B):
        # print(fingertip_T_batch[i])
        tool_repr = get_tool_repr(args, fingertip_T_batch[i], pkg='torch')
        init_pose_seq.append(tool_repr)

    init_pose_seqs = torch.stack(init_pose_seq).view(param_seqs.shape[0], param_seqs.shape[1], -1, 3)

    return init_pose_seqs


def params_to_init_pose(args, center, plan_params, param_seq, is_3d=False):
    ee_fingertip_T_mat = torch.FloatTensor(args.ee_fingertip_T_mat)

    init_pose_seq = []
    for params in param_seq:
        r, theta, phi1, phi2, grip_width = params

        center_x, center_y, center_z = center
        if is_3d:
            pos_x = center_x + (-r * torch.sin(phi1) + args.tool_center_z * torch.sin(phi2)) * 0.0
            pos_y = center_y + (-r * torch.sin(phi1) + args.tool_center_z * torch.sin(phi2)) * -1.0
            pos_z = center_z + r * torch.cos(phi1) + args.tool_center_z * torch.cos(phi2)

            ee_quat = qmult([0, 1, 0, 0], qmult(axangle2quat([0, 0, 1], 0 - np.pi / 4), 
                axangle2quat([1, 1, 0], phi2)))
        else:
            pos_x = center_x + (r * torch.sin(phi1) + args.tool_center_z * 0.0) * torch.sin(theta)
            pos_y = center_y + (r * torch.sin(phi1) + args.tool_center_z * 0.0) * torch.cos(theta)
            pos_z = center_z + r * torch.cos(phi1) + args.tool_center_z * 1.0

            ee_quat = qmult([0, 1, 0, 0], axangle2quat([0, 0, 1], theta - np.pi / 4))

        fingermid_pos = torch.FloatTensor([pos_x, pos_y, pos_z])
        ee_rot = quaternion_to_matrix(torch.FloatTensor(ee_quat))
        fingertip_mat = ee_rot @ ee_fingertip_T_mat[:3, :3]

        fingertip_T_list = []
        for k in range(len(args.tool_dim[args.env])):
            offset = torch.FloatTensor([0, (2 * k - 1) * (plan_params["init_grip"] / 2), 0])
            fingertip_pos = fingertip_mat @ offset + fingermid_pos
            fingertip_T = torch.cat((torch.cat((fingertip_mat, fingertip_pos.unsqueeze(1)), dim=1), 
                torch.FloatTensor([[0, 0, 0, 1]])), dim=0)
            fingertip_T_list.append(fingertip_T)

        fingertip_T_batch = torch.stack(fingertip_T_list)
        # print(fingertip_T_batch)
        tool_repr = get_tool_repr(args, fingertip_T_batch, pkg='torch')
        init_pose_seq.append(tool_repr)

    return torch.stack(init_pose_seq)


def param_seqs_to_actions(tool_params, param_seqs, step=1, is_3d=False):
    device = param_seqs.device
    B = param_seqs.shape[0] * param_seqs.shape[1]
    ps = param_seqs.view(B, -1)

    zero_pad = torch.zeros((B, 3), dtype=torch.float32, device=device)
    act_seq_list = []
    grip_rate = ((tool_params['init_grip'] - ps[:, -1]) / 2) / (tool_params['act_len'] / step)
    for _ in range(0, tool_params['act_len'], step):
        if is_3d:
            x = grip_rate * np.sin(np.pi / 2) 
            y = grip_rate * np.cos(np.pi / 2)
        else:
            x = grip_rate * torch.cos(ps[:, 1])
            y = -grip_rate * torch.sin(ps[:, 1])
        gripper_l_act = torch.stack((x, y, torch.zeros(B, dtype=torch.float32, device=device)), dim=-1)
        gripper_r_act = 0 - gripper_l_act
        act = torch.cat((gripper_l_act, zero_pad, gripper_r_act, zero_pad), dim=-1)
        act_seq_list.append(act)

    act_seqs = torch.stack(act_seq_list, dim=1).view(param_seqs.shape[0], param_seqs.shape[1], -1, 12)

    return act_seqs


def params_to_actions(tool_params, param_seq, step=1, is_3d=False):
    zero_pad = torch.zeros(3)
    act_seq = []
    for params in param_seq:
        actions = []
        r, theta, phi1, phi2, grip_width = params
        grip_rate = ((tool_params['init_grip'] - grip_width) / 2) / (tool_params['act_len'] / step)
        for _ in range(0, tool_params['act_len'], step):
            if is_3d:
                x = grip_rate * np.sin(np.pi / 2) 
                y = grip_rate * np.cos(np.pi / 2)
            else:
                x = grip_rate * torch.cos(theta)
                y = -grip_rate * torch.sin(theta)
            gripper_l_act = torch.cat([x.unsqueeze(0), y.unsqueeze(0), torch.zeros(1)])
            gripper_r_act = 0 - gripper_l_act
            act = torch.cat((gripper_l_act, zero_pad, gripper_r_act, zero_pad))
            actions.append(act)

        act_seq.append(torch.stack(actions))

    return torch.stack(act_seq)


def init_pose_to_params(init_pose_seq):
    # import pdb; pdb.set_trace()
    if not torch.is_tensor(init_pose_seq):
        init_pose_seq = torch.FloatTensor(init_pose_seq)

    mid_point = (init_pose_seq.shape[1] - 1) // 2
    mid_point_seq = (init_pose_seq[:, mid_point, :3] + init_pose_seq[:, mid_point, 7:10]) / 2

    rot_seq = torch.atan2(init_pose_seq[:, mid_point, 2] - mid_point_seq[:, 2], \
        init_pose_seq[:, mid_point, 0] - mid_point_seq[:, 0])

    a = init_pose_seq[:, 0, :3] - init_pose_seq[:, -1, :3]
    b = torch.FloatTensor([[0.0, 1.0, 0.0]]).expand(init_pose_seq.shape[0], -1)
    z_angle_seq = torch.acos((a * b).sum(dim=1) / (a.pow(2).sum(dim=1).pow(0.5) * b.pow(2).sum(dim=1).pow(0.5)))

    pi = torch.full(rot_seq.shape, np.pi)
    rot_seq = pi - rot_seq
    z_angle_seq = pi - z_angle_seq
    
    return mid_point_seq, rot_seq, z_angle_seq
