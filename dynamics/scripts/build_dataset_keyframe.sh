#!/usr/bin/env bash

python dynamics/scripts/build_dataset_keyframe.py \
	--stage perception \
	--tool_type gripper_sym_rod_robot_v2_surf_nocorr_full \
	--debug 0
