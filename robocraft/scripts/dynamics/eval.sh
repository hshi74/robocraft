#!/usr/bin/env bash

python eval.py \
	--stage dy \
	--data_type ngrip_fixed \
    --model_path dump/dump_ngrip_fixed/out_15-Feb-2022-22:25:15.259783/net_epoch_0_iter_100.pth \
	--n_rollout 5