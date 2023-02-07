#!/usr/bin/env bash

#SBATCH --job-name=robocraft3d
#SBATCH --account=viscam
#SBATCH --partition=svl
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/sailhome/hshi74/output/robocraft3d/%A.out

# kernprof -l
kernprof -l planning/control.py \
	--stage control \
	--tool_type $1 \
	--debug $2 \
	--target_shape_name $3 \
	--optim_algo $4 \
	--CEM_sample_size $5 \
	--control_loss_type $6 \
	--subtarget $7 \
	--close_loop $8 \
	--planner_type $9 \
	--max_n_actions ${10}
