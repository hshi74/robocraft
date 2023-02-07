#!/usr/bin/env bash

kernprof -l perception/sample.py \
	--stage perception \
	--tool_type gripper_sym_rod_robot_v1 \ # tool name
	--n_particles 300 \ # number of points in the sampled point cloud
	--surface_sample 1 # whether you want a surface point cloud of not
