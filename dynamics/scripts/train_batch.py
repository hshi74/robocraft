import os

tool_type="gripper_sym_rod_robot_v1_surf_nocorr_full_normal_keyframe=12"
data_type="keyframe"
debug=0
loss_type="chamfer_emd"
n_his=1
sequence_length=2
time_step=1
rigid_motion=1
attn=0
chamfer_weight=0.5
emd_weight=0.5
neighbor_radius=0.02
tool_neighbor_radius="0.02+0.02"
train_set_ratio=1.0


def main():
    for t in [1, 2, 3, 4]:
        param_str = f'{tool_type} {data_type} {debug} {loss_type} {n_his} {sequence_length} {t} {rigid_motion} {attn} {chamfer_weight} {emd_weight} {neighbor_radius} {tool_neighbor_radius} {train_set_ratio}'
        print(param_str)
        os.system(f'sbatch dynamics/scripts/train.sh {param_str}')

    for nr in [0.02, 0.025, 0.03, 0.035, 0.04]:        
        param_str = f'{tool_type} {data_type} {debug} {loss_type} {n_his} {sequence_length} {time_step} {rigid_motion} {attn} {chamfer_weight} {emd_weight} {nr} {tool_neighbor_radius} {train_set_ratio}'
        print(param_str)
        os.system(f'sbatch dynamics/scripts/train.sh {param_str}')

    for tnr in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]:        
        param_str = f'{tool_type} {data_type} {debug} {loss_type} {n_his} {sequence_length} {time_step} {rigid_motion} {attn} {chamfer_weight} {emd_weight} {neighbor_radius} {tnr}+{tnr} {train_set_ratio}'
        print(param_str)
        os.system(f'sbatch dynamics/scripts/train.sh {param_str}')


if __name__ == "__main__":
    main()
