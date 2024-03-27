# SemanticHuman-HD
The project for SemanticHuman-HD : High-Resolution Semantic Disentangled 3D Human Generation
---
## [Project Page](https://pengzheng0707.github.io/SemanticHuman-HD/) | [Paper](https://arxiv.org/pdf/2403.10166.pdf) | [arXiv](https://arxiv.org/abs/2403.10166)
***
## What is SemanticHuman-HD?
The main contributions of our human image systhesis model are listed as followed:
1. The first method to synthesize **semantic disentangled full-body** human image.
2. We propose a **3D-aware super-resolution module** which enable our project achieve 3D-aware image synthesis at $1024^2$ resolution.

## Pipeline
![Pipeline](/pic/pipeline_00.png)

## Qucik Start
### Requirements
* We finished our training and testing tasks on 4 NVIDIA A40 GPUs.
* 64-bit Python 3.8 and PyTorch 1.11.0 (or later). See [https://pytorch.org](https://pytorch.org) for PyTorch install instructions.
* CUDA toolkit 11.3 or later.
* pytorch3d. See [https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for install instructions.

Clone this repo:
```
git clone https://github.com/Alfonsoever/SemanticHuman-HD.git

cd SemanticHuman-HD
```

## Results

### View control
![3d1](/gif/3d1.gif)
![3d2](/gif/3d2.gif)
![3d3](/gif/3d3.gif)

### Pose control
![anim1](/gif/anim1.gif)
![anim2](/gif/anim2.gif)
![anim3](/gif/anim3.gif)

### Semantic disentanglement
![semantic_disentanglement](/pic/change_00.png)

### Out-of-domain image synthesis
![ood](/pic/ood_00.png)

### Garment generation
![garment](/pic/garment_00.png)

### Conditional image synthesis
![cond_image](/pic/cond_00.png)

### Semantic-aware interpolation
![Semantic_interpolation](/pic/interpolate_part_00.png)

### 3D garment interpolation
![garment_interpolation](/pic/interpolate_garment_00.png)

To see more details,please go to our [project page](https://pengzheng0707.github.io/SemanticHuman-HD/).

