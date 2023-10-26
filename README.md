# [ICCV 2023] Minimum Latency Deep Online Video Stabilization ([Paper](https://arxiv.org/pdf/2212.02073.pdf))

<h4 align="center">Zhuofan Zhang<sup>1,*</sup>, Zhen Liu<sup>2,*</sup>,  Ping Tan<sup>3</sup>,  Bing Zeng<sup>1</sup>,  Shuaicheng Liu<sup>1,2,†</sup></center>
<h4 align="center">1. University of Electronic Science and Technology of China,  2. Megvii Research</center>
<h4 align="center">3. The Hong Kong University of Science and Technology</center>
<h6 align="center">*Equal contribution,  †Corresponding author</center>


## Abstract

We present a novel camera path optimization framework for the task of online video stabilization. Typically, a stabilization pipeline consists of three steps: motion estimating, path smoothing, and novel view rendering. Most previous methods concentrate on motion estimation, proposing various global or local motion models. In contrast, path optimization receives relatively less attention, especially in the important online setting, where no future frames are available. In this work, we adopt recent off-the-shelf high-quality deep motion models for the motion estimation to recover the camera trajectory and focus on the latter two steps. Our network takes a short 2D camera path in a sliding window as input and outputs the stabilizing warp field of the last frame in the window, which warps the coming frame to its stabilized position. A hybrid loss is well-defined to constrain the spatial and temporal consistency. In addition, we build a motion dataset that contains stable and unstable motion pairs for the training. Extensive experiments demonstrate that our approach significantly outperforms state-of-the-art online methods both qualitatively and quantitatively and achieves comparable performance to offline methods.

## Pipeline

![pipeline](https://github.com/liuzhen03/NNDVS/assets/18542006/d8753a30-4aa1-4bb9-92aa-fc003a658578)

## MotionStab Dataset

The `MotionStab` dataset and the synthesized stable/unstable videos can be download from [Google Drive](https://drive.google.com/drive/folders/1zKK6Ffp8TKbMptx4kyCuHNPr3JBQND5R?usp=drive_link). The dataset is organized as follow:

```
MotionStab
|--Regular
|  |--0-fi.npy
|  |--0-bi.npy
|  |--0-unstab.mp4
|  |--0-stab.mp4
|  |--...
|--QuickRotation
|  |--0-fi.npy
|  |--0-bi.npy
|  |--0-unstab.mp4
|  |--0-stab.mp4
|  |--...
|--Crowd
|  |--0-fi.npy
|  |--0-bi.npy
|  |--0-unstab.mp4
|  |--0-stab.mp4
|  |--...
...
```

For each synthesized video, `xx-fi.npy`, `xx-bi.npy`, `xx-stab.mp4`, `xx-unstab.mp4` are inter-frame motions, ground truth warp fields, the unstable video, and the stable video, respectively.

## Usage

### Requirements

* Python 3.7.13
* PyTorch 1.9.0
* Torchvision 0.10.0
* CUDA 10.2 on Ubuntu 18.04

Install the require dependencies:

```bash
conda create -n nndvs python=3.7
conda activate nndvs
pip install -r requirements.txt
```

### Evaluation

1. Download the reorganized `NUS` dataset from [Google Drive](https://drive.google.com/drive/folders/1Pm-6G5-Lrm9SGdjDcRYHoqDUdvn58PfI?usp=drive_link) and place it in the `./data` folder.

2. Conduct full evaluation by running:

   ```
   bash eval_nus.sh
   ```

## Citation

If you find this work helpful, please cite our paper:

```
@InProceedings{Zhang_2023_ICCV,
    author    = {Zhang, Zhuofan and Liu, Zhen and Tan, Ping and Zeng, Bing and Liu, Shuaicheng},
    title     = {Minimum Latency Deep Online Video Stabilization},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {23030-23039}
}
```

## Contact

If you have any questions, feel free to contact Zhen Liu at liuzhen03@megvii.com.
