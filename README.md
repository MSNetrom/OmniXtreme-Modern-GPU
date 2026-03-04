
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

### 0. Install Micromamba
```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

### 1. Create a GPU-ready Micromamba environment
```bash
micromamba create -f environment.yml
micromamba activate omnixtreme
```

### 3. Set motion path
Download the curated subset of our motion data from [this link](https://drive.google.com/file/d/1-LXEmUfW80BXYQ0tAZx341PzY8u14Z9Q/view?usp=sharing) and move it to the policy directory `policy/`. 

### 4. Run policy
```bash
python deploy_mujoco.py
```
