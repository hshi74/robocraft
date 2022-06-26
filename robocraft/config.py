import argparse
import numpy as np
from datetime import datetime

### build arguments
parser = argparse.ArgumentParser()
parser.add_argument('--env', default='Gripper')
parser.add_argument('--stage', default='dy', help="dy: dynamics model; control")
parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--random_seed', type=int, default=42)

parser.add_argument('--time_step', type=int, default=119)
parser.add_argument('--dt', type=float, default=1. / 60.)
parser.add_argument('--n_instance', type=int, default=1)

parser.add_argument('--nf_relation', type=int, default=150)
parser.add_argument('--nf_particle', type=int, default=150)
parser.add_argument('--nf_pos', type=int, default=150)
parser.add_argument('--nf_memory', type=int, default=150)
parser.add_argument('--mem_nlayer', type=int, default=2)
parser.add_argument('--nf_effect', type=int, default=150)

parser.add_argument('--stdreg', type=int, default=0)
parser.add_argument('--stdreg_weight', type=float, default=0.0)
parser.add_argument('--matched_motion', type=int, default=0)
parser.add_argument('--matched_motion_weight', type=float, default=0.0)

parser.add_argument('--valid', type=int, default=0)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--verbose_data', type=int, default=0)
parser.add_argument('--verbose_model', type=int, default=0)
parser.add_argument('--eps', type=float, default=1e-6)

# file paths
parser.add_argument('--outf_eval', default='')
parser.add_argument('--controlf', default='')
parser.add_argument('--outf_new', default='')
parser.add_argument('--gripperf', default='../simulator/plb/envs/gripper_fixed.yml')

# for ablation study
parser.add_argument('--neighbor_radius', type=float, default=0.05)
parser.add_argument('--gripper_extra_neighbor_radius', type=float, default=0.015)
parser.add_argument('--neighbor_k', type=float, default=20)

# shape state:
# [x, y, z, x_last, y_last, z_last, quat(4), quat_last(4)]
parser.add_argument('--shape_state_dim', type=int, default=14)

# object attributes:
parser.add_argument('--attr_dim', type=int, default=3)
# object state:
parser.add_argument('--state_dim', type=int, default=3)
# relation attr:
parser.add_argument('--relation_dim', type=int, default=0)

# physics parameter
parser.add_argument('--physics_param_range', type=float, nargs=2, default=None)

# width and height for storing vision
parser.add_argument('--vis_width', type=int, default=160)
parser.add_argument('--vis_height', type=int, default=120)


'''
train
'''
parser.add_argument('--data_type', type=str, default='none')
parser.add_argument('--gt_particles', type=int, default=0)
parser.add_argument('--shape_aug', type=int, default=1)

parser.add_argument('--loss_type', type=str, default='emd_chamfer_h')
parser.add_argument('--h_weight', type=float, default=0.0)
parser.add_argument('--emd_weight', type=float, default=0.9)
parser.add_argument('--chamfer_weight', type=float, default=0.1)
parser.add_argument('--p_rigid', type=float, default=1.0)

# use a flexible number of frames for each training iteration
parser.add_argument('--n_his', type=int, default=4)
parser.add_argument('--sequence_length', type=int, default=6)

parser.add_argument('--n_rollout', type=int, default=0)
parser.add_argument('--train_valid_ratio', type=float, default=0.9)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--log_per_iter', type=int, default=100)
parser.add_argument('--ckp_per_iter', type=int, default=1000)

parser.add_argument('--n_epoch', type=int, default=100) # 100 FOR TEST, *1000* 
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--optimizer', default='Adam', help='Adam|SGD')
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=4)

# data generation
parser.add_argument('--gen_data', type=int, default=0)
parser.add_argument('--gen_stat', type=int, default=0)
parser.add_argument('--gen_vision', type=int, default=0)

parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--resume_epoch', type=int, default=0)
parser.add_argument('--resume_iter', type=int, default=0)

# data augmentation
parser.add_argument('--augment_ratio', type=float, default=0.05)


# visualization flog
parser.add_argument('--pyflex', type=int, default=1)
parser.add_argument('--vis', type=str, default='plt')


'''
control
'''
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--opt_algo', type=str, default='max')
parser.add_argument('--control_algo', type=str, default='fix')
parser.add_argument('--predict_horizon', type=int, default=2)
parser.add_argument('--shooting_size', type=int, default=200)
parser.add_argument('--shooting_batch_size', type=int, default=4)
parser.add_argument('--reward_type', type=str, default='emd_chamfer_h')
parser.add_argument('--use_sim', type=int, default=0)
parser.add_argument('--gt_action', type=int, default=0)
parser.add_argument('--gt_state_goal', type=int, default=0)
parser.add_argument('--subgoal', type=int, default=0)
parser.add_argument('--correction', type=int, default=0)
parser.add_argument('--n_grips', type=int, default=3)
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--shape_type', type=str, default='')
parser.add_argument('--goal_shape_name', type=str, default='')
parser.add_argument('--outf_control', default='')

'''
eval
'''
parser.add_argument('--eval_epoch', type=int, default=-1, help='pretrained model')
parser.add_argument('--eval_iter', type=int, default=-1, help='pretrained model')
parser.add_argument('--eval_set', default='train')

### only useful for rl
parser.add_argument("--algo", type=str, default='sac')
parser.add_argument("--env_name", type=str, default="gripper_fixed-v1")
parser.add_argument("--path", type=str, default='./tmp')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_steps", type=int, default=None)

# differentiable physics parameters
parser.add_argument("--rllr", type=float, default=0.1)
parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'Momentum'])

def gen_args():
    args = parser.parse_args()

    if args.env != 'Gripper':
        raise AssertionError("Unsupported env")

    args.data_names = ['positions', 'shape_quats', 'scene_params']
    args.physics_param_range = (-5., -5.)

    # path to data
    args.dataf = f'data/data_{args.data_type}'
    # path to output
    args.outf =  f'dump/dump_{args.data_type}/out_{datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")}'

    args.mean_p = np.array([0.50932539, 0.11348496, 0.49837578])
    args.std_p = np.array([0.06474939, 0.04888084, 0.05906044])

    args.mean_d = np.array([-0.00284736, 0.00286124, -0.00130389])
    args.std_d = np.array([0.01755744, 0.01663332, 0.01677678])

    return args
