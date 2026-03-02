
# OmniXtreme: Breaking the Generality Barrier in High-Dynamic Humanoid Control

[![arXiv](https://img.shields.io/badge/arXiv-2602.23843-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.23843)
[![paper](https://img.shields.io/badge/Paper-PDF-red?logo=adobeacrobatreader)](https://arxiv.org/pdf/2602.23843)
[![github](https://img.shields.io/badge/Project_Page-OmniXtreme-green)](https://extreme-humanoid.github.io/)


## Overview

This repository contains the official implementation of **OmniXtreme**, a unified policy framework for high-dynamic humanoid motion tracking.

We release:

- ✅ Paper and demonstration videos  
- ✅ Checkpoints  
- ✅ Sim-to-sim evaluation code  

The provided pretrained policy under `policy/` is trained using the **OmniXtreme framework** on high-dynamic motions.

---

## Planned Releases

The following components are under consideration for future open-source release:

- Flow matching base policy training and inference code  
- Residual post-training and inference code  
- C++ real-world deployment code  

---

## Installation
### 1. Create conda environment and set environment variable
```
git clone https://github.com/Perkins729/OmniXtreme.git
cd OmniXtreme
conda create -n omnixtreme python=3.8
conda activate omnixtreme
conda install -c conda-forge cudnn=8
pip install -r requirements.txt
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```





### 2. Set motion path
Download the curated subset of our motion data from [this link](https://drive.google.com/file/d/1-LXEmUfW80BXYQ0tAZx341PzY8u14Z9Q/view?usp=sharing) and move it to the policy directory `policy/`. 

### 3.Run policy
```
python deploy_mujoco.py
```
