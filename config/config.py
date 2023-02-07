import argparse
import copy
import numpy as np
import os
import sys
import torch
import yaml

from utils.data_utils import get_square, load_tool_repr
from datetime import datetime

# build arguments
parser = argparse.ArgumentParser()

# accessible arguments through bash scripts
########## General ##########
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--env', type=str, default='gripper_sym_rod')
parser.add_argument('--random_seed', type=int, default=42)
# ['perception', 'dy(namics)', 'control']
parser.add_argument('--stage', default='dy')
parser.add_argument('--tool_type', type=str, default='gripper_sym_rod_robot_v1_surf_nocorr_full')
parser.add_argument('--use_gpu', type=int, default=1)
parser.add_argument('--n_particles', type=int, default=300)


########## Perception ##########
# ==================== TUNE at PERCEPTION ==================== #
parser.add_argument('--surface_sample', type=int, default=1)
parser.add_argument('--correspondance', type=int, default=0)
# ==================== TUNE at PERCEPTION ==================== #


########## Dynamics ##########
##### Train #####
# ==================== TUNE at TRAINING ==================== #
parser.add_argument('--batch_norm', type=int, default=0)
parser.add_argument('--train_set_ratio', type=float, default=1.0)
parser.add_argument('--valid', type=int, default=1)
parser.add_argument('--data_type', type=str, default='gt')
parser.add_argument('--loss_type', type=str, default='chamfer_emd')
parser.add_argument('--rigid_motion', type=int, default=0)
parser.add_argument('--attn', type=int, default=0)
parser.add_argument('--full_repr', type=int, default=1)

parser.add_argument('--neighbor_radius', type=float, default=0.01)
parser.add_argument('--tool_neighbor_radius', type=str, default='0.01+0.01')
parser.add_argument('--motion_bound', type=float, default=0.005)
parser.add_argument('--sequence_length', type=int, default=3)
parser.add_argument('--chamfer_weight', type=float, default=0.5)
parser.add_argument('--emd_weight', type=float, default=0.5)
parser.add_argument('--h_weight', type=float, default=0.0)
parser.add_argument('--loss_ord', type=int, default=2)
parser.add_argument('--dcd_alpha', type=int, default=50)
parser.add_argument('--dcd_n_lambda', type=float, default=0.4)
# ==================== TUNE at TRAINING ==================== #

parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--plateau_epoch_size', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--optimizer', default='Adam')

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--dy_model_path', type=str, default='')
parser.add_argument('--resume_path', type=str, default='')
parser.add_argument('--ckp_per_iter', type=int, default=10000)

##### GNN #####
parser.add_argument('--n_his', type=int, default=4)
parser.add_argument('--p_rigid', type=float, default=1.0)
parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--time_step', type=int, default=1)

##### Eval #####
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--eval_epoch', type=int, default=-1)
parser.add_argument('--eval_iter', type=int, default=-1)
parser.add_argument('--n_rollout', type=int, default=0)


########## Control ########## 
parser.add_argument('--close_loop', type=int, default=1)
parser.add_argument('--tool_model_name', type=str, default='default')
parser.add_argument('--active_tool_list', type=str, default='default')
parser.add_argument('--gt_action', type=int, default=0)
parser.add_argument('--gt_state_goal', type=int, default=0)
parser.add_argument('--max_n_actions', type=int, default=2)
parser.add_argument('--optim_algo', type=str, default='max')
parser.add_argument('--control_loss_type', type=str, default='chamfer')
parser.add_argument('--subtarget', type=int, default=0)
parser.add_argument('--target_shape_name', type=str, default='')
parser.add_argument('--cls_type', type=str, default='pcd')
parser.add_argument('--planner_type', type=str, default='gnn')

parser.add_argument('--control_batch_size', type=int, default=8)
# parser.add_argument('--RS_sample_size', type=int, default=128)
parser.add_argument('--RS_elite_size_per_act', type=int, default=3)
parser.add_argument('--CEM_sample_size', type=int, default=40)
parser.add_argument('--CEM_elite_size', type=float, default=10)
parser.add_argument('--CEM_sample_iter', type=int, default=20)
parser.add_argument('--CEM_decay_factor', type=float, default=0.5)


########## RL ########## 
parser.add_argument("--rl_algo", type=str, default='sac')
parser.add_argument("--rl_env_name", type=str, default="gripper_fixed-v1")
parser.add_argument("--rl_num_steps", type=int, default=None)
parser.add_argument("--rl_optim", type=str, default='Adam', choices=['Adam', 'Momentum'])
parser.add_argument("--rl_path", type=str, default='./tmp')
parser.add_argument("--rl_seed", type=int, default=0)


# precoded arguments
def gen_args():
    args = parser.parse_args()

    args.data_names = ['positions', 'shape_quats', 'scene_params']
    args.physics_param_range = (-5., -5.)
    args.scene_params = np.array([1, 1, 0]) # n_instance, gravity, draw_mesh

    if 'robot' in args.tool_type:
        args.env = args.tool_type.split('_robot')[0]
    else:
        args.env = args.tool_type.split('_sim')[0]

    if 'full' in args.tool_type:
        args.full_repr = 1

    args = gen_args_env(args)
    args = gen_args_pipeline(args)

    return args


def gen_args_env(args):
    ##### path ######
    # training data
    args.dy_data_path = f'data/{args.data_type}/data_{args.tool_type}'
    # traing output
    args.dy_out_path =  f'dump/dynamics/dump_{args.tool_type}'
    # ROS package
    args.ros_pkg_path = "/scr/hshi74/catkin_ws/src/deformable_ros"
    # args.ros_pkg_path = "/home/haochen/catkin_ws/src/deformable_ros"
    # sim config file
    # args.sim_config_path = f'config/taichi_env/{args.env}.yml'
    # tool models
    args.tool_geom_path = "geometries/tools"
    # tool repr models
    args.tool_repr_path = "geometries/reprs"

    args.data_time_step = 1

    ##### camera ######
    args.depth_optical_frame_pose = [0, 0, 0, 0.5, -0.5, 0.5, -0.5]
    if not 'viscam' in os.path.realpath(sys.argv[0]):
        with open(os.path.join(args.ros_pkg_path, 'env', 'camera_pose_world.yml'), 'r') as f:
            args.cam_pose_dict = yaml.load(f, Loader=yaml.FullLoader)

    ##### robot #####
    # if args.stage != 'dy':
    args.ee_fingertip_T_mat = np.array([[0.707, 0.707, 0, 0], [-0.707, 0.707, 0, 0], [0, 0, 1, 0.1034], [0, 0, 0, 1]])
    # 5cm: 0.042192, 7cm: 0.052192, 8cm: 0.057192
    args.tool_center_z = 0.06472

    mid_point_sim = np.array([0.5, 0.1, 0.5])
    # robocraft
    mid_point_robot = np.array([0.43, -0.01, 0.1])

    if 'robot' in args.tool_type:
        args.mid_point = mid_point_robot
        args.axes = [0, 1, 2]
    else:
        args.mid_point = mid_point_sim
        args.axes = [0, 2, 1]

    ##### floor #####
    args.floor_dim = 9
    if 'robot' in args.tool_type:
        args.floor_unit_size = 0.05 # distance between two neighbor dots
        args.floor_pos = np.array([0.43, -0.01, 0.08])
    else:        
        args.floor_unit_size = 0.25 
        args.floor_pos = np.array([0.5, 0, 0.5])
    
    args.floor_state = get_square(args.floor_pos, args.floor_unit_size, args.floor_dim, args.axes[2])
    args.floor_normals = np.tile([0.0, 0.0, 1.0], (args.floor_dim, 1))

    ##### tools #####
    args.tool_geom_mapping = {
        'gripper_sym_rod': ['gripper_l_8', 'gripper_r_8'], # ['gripper_l', 'gripper_r'],
    }

    args.tool_sim_primitive_mapping = {
        'gripper_sym_rod': [0, 1],
    }

    args.tool_action_space_size = {
        'gripper_sym_rod': 4,
    }

    if args.full_repr:
        args.tool_dim = {
            'gripper_sym_rod': [182, 182] # [92, 92],
        }
    else:
        raise NotImplementedError

    args.tool_neighbor_radius = [float(x) for x in args.tool_neighbor_radius.split('+')]

    args.tool_full_repr_dict = load_tool_repr(args)

    return args


def gen_args_pipeline(args):
    ##### dynamics #####
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")

    if 'surf' in args.tool_type:
        args.surface_sample = 1

    if args.tool_type.split('_')[-1].isnumeric():
        args.n_particles = int(args.tool_type.split('_')[-1])

    # if 'gripper' in args.env:
    #     args.rigid_motion = 1
    # else:
    #     args.rigid_motion = 0

    if 'normal' in args.tool_type:
        args.state_dim = 6
        args.mean_p = np.array([*args.mid_point, 0.0, 0.0, 0.0])
        # args.mean_p[args.axes[-1]] = 0.02
        args.std_p = np.array([0.017, 0.017, 0.01, 1.0, 1.0, 1.0])
    else:
        args.state_dim = 3
        args.mean_p = np.array([*args.mid_point])
        # args.mean_p[args.axes[-1]] = 0.02
        args.std_p = np.array([0.017, 0.017, 0.01])

    args.mean_d = np.array([1e-05, 1e-05, 1e-05])
    args.std_d = np.array([0.0003, 0.0003, 0.0003])

    args.n_instance = 1
    args.nf_relation = 150
    args.nf_particle = 150
    args.nf_pos = 150
    args.nf_memory = 150
    args.mem_nlayer = 2
    args.nf_effect = 150

    return args


def update_dy_args(args, dy_args_dict):
    args.surface_sample = dy_args_dict['surface_sample']
    args.rigid_motion = dy_args_dict['rigid_motion']
    args.attn = dy_args_dict['attn']
    args.neighbor_radius = dy_args_dict['neighbor_radius']
    args.full_repr = dy_args_dict['full_repr']
    args.tool_dim = dy_args_dict['tool_dim']
    args.tool_type = dy_args_dict['tool_type']
    args.tool_neighbor_radius = dy_args_dict['tool_neighbor_radius']
    args.data_type = dy_args_dict['data_type']
    args.n_his = dy_args_dict['n_his']
    args.time_step = dy_args_dict['time_step']
    args.state_dim = dy_args_dict['state_dim']
    args.mean_p = dy_args_dict['mean_p']
    args.std_p = dy_args_dict['std_p']

    return args


def print_args(args):
    for key, value in dict(sorted(args.__dict__.items())).items():
        if not key in ['floor_state', 'tool_center', 'tool_geom_mapping', 'tool_neighbor_radius_dict', 'tool_full_repr_dict']:
            print(f'{key}: {value}')
