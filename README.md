# distortion-free-wide-angle.pytorch

Corrects perspective aberrations in wide-angle portraits. Implementation of the paper [Distortion-Free Wide-Angle Portraits on Camera Phones](https://people.csail.mit.edu/yichangshih/wide_angle_portrait/) [1] | Course project for [15-663 Computational Photography](http://graphics.cs.cmu.edu/courses/15-463/).

## Introduction

![Demo](https://raw.githubusercontent.com/yzhq97/distortion-free-wide-angle.pytorch/main/demo.png "Demo")

Wide-angle portrait mode has been extensively used in taking selﬁes because it enables the camera to include more people and faces in the photo. 
However, lenses with large FOV tend to create artifacts in the photos taken.
This is caused by the nature of perspective projection, which projects the surrounding world onto a ﬂat image.
Motivated by this issue, the authors proposed an automatic algorithm to do post-processing on-site immediately after photos are taken. Their algorithm reverses perspective distortion in wide-angle portraits, so that everyone in the photo looks natural and real.

[Full Project Report](https://github.com/yzhq97/distortion-free-wide-angle.pytorch/blob/main/report.pdf)

## Environment

* Python >= 3.6
* PyTorch >= 1.6 and matching torchvision

Recommended configuration with Anaconda:
1. Install pytorch `conda install pytorch==1.6.0 torchvision -c pytorch`.
2. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).
3. Install other dependencies `pip install -r requirements.txt`.

## Usage

Basic usage:
`python src/main.py --file <path-to-photo>`

Advanced usage:
```
python src/main.py --file FILE [--num_iter NUM_ITER] [--lr LR] [--Q Q]
                   [--mesh_ds_ratio MESH_DS_RATIO] [--naive NAIVE]
                   [--face_energy FACE_ENERGY] [--similarity SIMILARITY]
                   [--line_bending LINE_BENDING] [--regularization REGULARIZATION]
                   [--boundary_constraint BOUNDARY_CONSTRAINT]

--file                path to photo
--num_iter            number of optimization steps
--lr                  learning rate
--Q                   number of padding vertices
--mesh_ds_ratio       the pixel-to-vertex ratio
--naive               if set True, perform naive orthographic correction
--face_energy         weight of the face energy term
--similarity          weight of similarity tranformation constraint
--line_bending        weight of the line bending term
--regularization      weight of the regularization term
--boundary_constraint weight of the mesh boundary constraint
```

Please refer to our project report for details of the algorithm.

## Project Structure

```
.                             run all commands under this directory
├── data                      put images under this directory
└── src                       source code directory
    ├── data.py               implements dataloader
    ├── energy.py             definition of the optimized energy function
    ├── main.py               main script
    ├── perception.py         implements subject detection and segmentation
    ├── stereographic.py      implements naive stereographic projection
    └── visualization.py      implements visualization tools
```

## References and Acknowledgement

[1] YiChang Shih, Wei-Sheng Lai, and Chia-Kai Liang. Distortion-free wide-angle portraits on camera phones. ACM Trans. Graph., 38(4), July 2019.

We sincerely thank the authors for sharing their ideas, data and insights.
