
# RoboCraft: Learning to See, Simulate, and Shape Elasto-Plastic Object with Graph Networks

## DEPRECATED: Please see the dev branch for an improved version of the RoboCraft codebase, especially if you're interested in real-world experiments! I'm actively maintaining this codebase, feel free to open an issue or shoot me an email (hshi74@stanford.edu) if you have any questions!

## Overview

This is the codebase of [RoboCraft](http://hxu.rocks/robocraft/) in the simulator.

**[Project Page](http://hxu.rocks/robocraft/) |  [Paper](https://arxiv.org/pdf/2205.02909.pdf)**

<img src="images/robocraft.gif" width="600">

## Prerequisites
- Linux or macOS (Tested on Ubuntu 20.04)
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- Conda

## Getting Started

### Setup
```bash
# clone the repo
git clone https://github.com/hshi74/RoboCraft.git
cd RoboCraft

# create the conda environment
conda env create -f robocraft.yml
conda activate robocraft

# install requirements for the simulator
cd simulator
pip install -e .
```

### Generate Data
- Run all the blocks in `simulator/plb/algorithms/test_tasks.ipynb`. Note that it is easier to use ipython notebook when dealing with Taichi env for fast materialization.
- You can control the number of videos to generate by changing the variable `n_vid`. The default is 5 for the purpose of debugging. We used 50 in the paper. 

### Sample Particles
- `cd simulator/plb/algorithms`
- Go to line 598 in `sample_data.py` and replace the string with the output folder name you just generated in `simulator/dataset`
- Run `python sample_data.py`. Note that this step may take a while.

### Build the dataset for GNN
- You will need to remove the old dataset if you want to update the dataset
```bash
cd ../../../robocraft
bash scripts/utils/move_data.sh ngrip_fixed sample_ngrip_fixed_[timestamp of the folder you just generated]
```

### Train Dynamics Model
`bash scripts/dynamics/train.sh`

### Planning with the Learned Model
- Go to line 6 in `robocraft/scripts/control/control.sh` and change the model path to the path to the checkpoint you just trained.
- `bash scripts/control/control.sh`

## Code structure
- The simulator folder contains the simulation environment we used for data collection and particle sampling. 
- The robocraft folder contains the code for learning the GNN and planning.

## Citation
If you use the codebase in your research, please cite:
```
@article{shi2022robocraft,
  title={RoboCraft: Learning to See, Simulate, and Shape Elasto-Plastic Objects with Graph Networks},
  author={Shi, Haochen and Xu, Huazhe and Huang, Zhiao and Li, Yunzhu and Wu, Jiajun},
  journal={arXiv preprint arXiv:2205.02909},
  year={2022}
}
```
