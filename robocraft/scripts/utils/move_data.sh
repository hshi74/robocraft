#!/usr/bin/env bash

# Make sure you're in the robocraft dir before running the following commands

if [[ $# -eq 0 ]] ; then
    echo 'There are 2 arguments required: 1. data_type (e.g. ngrip_fixed) 2. sampling_result_dir (e.g. sample_ngrip_fixed_14-Feb-2022-21:24:27.516157)'
    exit 0
fi

training_set_dir="./data/data_$1/train"
if [ ! -d "$training_set_dir" ] || [ -z "$(ls -A $training_set_dir)" ]; then
    mkdir -p ./data/data_$1/train
    cp -r ../simulator/dataset/$2/* $training_set_dir
else
    echo 'The training set already exists!'
fi

valid_set_dir="./data/data_$1/valid"
training_set_size=`find $training_set_dir/* -maxdepth 0 -type d | wc -l`
valid_set_size=$((training_set_size / 5))

echo "Training set size: $training_set_size"

if [ ! -d "$valid_set_dir" ] || [ -z "$(ls -A $valid_set_dir)" ]; then
    mkdir -p ./data/data_$1/valid
    for i in $(seq -f "%03g" 0 $((valid_set_size - 1))); do
        cp -r $training_set_dir/$i $valid_set_dir
    done
else
    echo 'The valid set already exists!'
fi

echo "Valid set size: $valid_set_size"