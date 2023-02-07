
# RoboCraft: Learning to See, Simulate, and Shape Elasto-Plastic Object with Graph Networks

## IMPORTANT: This is an improved version of the original RoboCraft codebase. This codebase mainly focuses on real-world experiments but could transfer to simulation environments with some refactorizations as well.

<!-- ## [The controller codebase of RoboCraft](https://github.com/hshi74/deformable_ros)
The controller is based on ROS Noetic and [Polymetis](https://facebookresearch.github.io/fairo/polymetis/). It's especially useful if you're working with a Franka Panda robot arm and a Franka hand gripper. -->

## Overview

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

# install pytorch
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

### Generate Data
<!-- - See [the controller codebase of RoboCraft](https://github.com/hshi74/deformable_ros) -->
- We will release the controller codebase soon. Stay tuned!

### For all the following bash or python scripts, you will need to modify certain hyperparameters (like directories) before you run them.

### Sample particles and build the dataset for the GNN
1. `bash perception/scripts/run_sample.sh`.
1. (Optional) There could be a tiny portion of problematic datapoints. If you want to manually check the sampling results and remove the problematic ones, 
    1. Run `perception/scripts/inspect_perception.sh` to move all the visualizations into the same folder for your convenience
    1. Manually go through all the videos and type the indices of problematic videos into `dump/perception/inspect/inspect.txt`
    1. Run `perception/scripts/clean_perception.sh` to remove all the problemtaic ones.
1. Run `percetion/scripts/make_dataset.py` to build the dataset for the GNN

### Train Dynamics Model
`bash dynamics/scripts/run_train.sh`

### Planning with the Learned Model
`bash scripts/control/run_control.sh`

## Code structure
- `config/`: config files for perception, dyanmics, planning, and simulation
- `dynamics/`: scripts to train and evaluate the GNN.
- `geometries/`: the STL files for tools and assets and surface point cloud representations for tools (in the `.npy` files)
- `models/`: a GNN checkpoint trained by us and its configurations (in the `.npy` file)
- `perception/`: the perception module of RoboCraft
- `planning/`: the planning module of RoboCraft
- `simulator/`: the simulation environment [PlasticineLab](https://github.com/hzaskywalker/PlasticineLab)
- `target_shapes/`: point clouds of some target shapes
- `utils/`: utility and visualization functions

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
