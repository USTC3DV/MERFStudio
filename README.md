# City-on-Web

<img src="assets/campus_short.gif" height="400"/> <img src="assets/mc_02.gif" height="400"/>

Ever wanted to render a (large-scale) NeRF scene in real-time on the web? Welcome to try our code.

This repository contains the official implementation of city-on-web and the PyTorch implementation of MERF, based on the nerfstudio framework.

> __City-on-Web:Real-time Neural Rendering of Large-scale Scenes on the Web__  
> [Kaiwen Song](SA21001046@mail.ustc.edu.cn), Xiaoyi Zeng, Chenqu Ren, [Juyong Zhang](http://staff.ustc.edu.cn/~juyong/index.html)  
> _European Conference on Computer Vision_ (__ECCV__)  
> [Project page](https://ustc3dv.github.io/City-on-Web/)&nbsp;/ [Paper](https://arxiv.org/abs/2312.16457)&nbsp;/ [Twitter](https://x.com/_akhaliq/status/1740589735024975967)&nbsp;

## Pipeline
<img src="https://raw.githubusercontent.com/kevin2000A/City-on-Web/main/assets/Pipeline.png"/> 

During the training phase, we uniformly partition the scene and reconstruct it at the finest LOD. To ensure 3D consistency, we use a resource-independent block-based volume rendering strategy. For LOD generation, we downsample virtual grid points and retrain a coarser model. This approach supports subsequent real-time rendering by facilitating the dynamic loading of rendering resources.


## Core Idea of block-based volume rendering
<!-- <video src="assets/diff.mp4"/>  -->
Unlike existing block-based methods that require the resources of all blocks to be loaded simultaneously for rendering, our strategy can be rendered independently using its own texture in its own shader. _This novel strategy supports asynchronous resource loading and independent rendering, which allows the strategy to be applied to other resource-independent environments, paving the way for further research and applications_. We are very pleased to see this strategy being applied to other large-scale scene reconstruction scenarios, such as multi-GPU parallel or distributed training of NeRF, and even  Gaussian splatting.


## Installation: Setup the environment

The code has been tested with Nvidia A100.

### Clone this repository

```shell
git clone --recursive https://github.com/Totoro97/f2-nerf.git
cd merfstudio
```

### Create environment
We recommend using conda to manage dependencies. Make sure to install Conda before proceeding.

```shell
conda create --name merfstudio -y python=3.8
conda activate merfstudio
```
### Dependencies
Install PyTorch with CUDA (this repo has been tested with CUDA 11.8) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
`cuda-toolkit` is required for building `tiny-cuda-nn`.

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### Install Nerfstudio
Our code has been tested with nerfstudio==0.3.3. There may be issues with newer versions of nerfstudio, but we will update it later to ensure compatibility with the latest version of nerfstudio.
```bash
cd nerfstudio
pip install -e .
```

### Install MERF and City-on-web
You can easily use the following command to install merf.
```bash
cd ..
pip install -e .
```

and city-on-web. Notably, the installation of city-on-web requires the prior installation of merf.

```bash
cd block-merf
pip install -e .
```

## Data preparation
Our data format requirements follow the instant-ngp convention.

### Mip-NeRF 360 Dataset
To download the mip-NeRF 360 dataset, visit the [official page](https://jonbarron.info/mipnerf360/). Then use `ns-process-data` to generate the `transforms.json` file.

```
ns-process-data images --data path/to/data --skip-colmap 
```


### Matrix City
To download the Matrix City dataset, visit the [official page](https://city-super.github.io/matrixcity/). You can opt to download the "small city"  dataset to test your algorithm. This dataset follows the instant-ngp convention, so no preprocessing is required.

### Custom Data

We highly recommend using [Metashape](https://www.agisoft.com/) to obtain camera poses from multi-view images. Then, use their [script](https://github.com/agisoft-llc/metashape-scripts/blob/master/src/export_for_gaussian_splatting.py) to convert camera poses to the COLMAP convention. Alternatively, you can use [COLMAP](https://github.com/colmap/colmap) to obtain the camera poses. After obtaining the data in COLMAP format, use `ns-process-data` to generate the `transforms.json` file.

```
ns-process-data images --data path/to/data --skip-colmap 
```


Our  loaders expect the following dataset structure in the source path location:

```
<location>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse(optionally)
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
|---transforms.json
```


## MERF Workflow
Our implementation of MERF is primarily based on the [official implementation](https://github.com/google-research/google-research/tree/master/merf).

### Training
To reconstruct a small scene like mip-nerf 360, simply use:

```
ns-trian merf --data path/to/data
```

### Baking 
To bake a reconstructed MERF scene, simply use the following command and load config from the config in the training output folder:
```
ns-baking --load-config path/to/output/config
```


### Real-Time Rendering 
Our results are compatible with the official MERF renderer. You can follow the [official guidance]() and place the baking results in the `webviewer` folder.
```
cd webviewer
mkdir -p third_party
curl https://unpkg.com/three@0.113.1/build/three.js --output third_party/three.js
curl https://unpkg.com/three@0.113.1/examples/js/controls/OrbitControls.js --output third_party/OrbitControls.js
curl https://unpkg.com/three@0.113.1/examples/js/controls/PointerLockControls.js --output third_party/PointerLockControls.js
curl https://unpkg.com/png-js@1.0.0/zlib.js --output third_party/zlib.js
curl https://unpkg.com/png-js@1.0.0/png.js --output third_party/png.js
curl https://unpkg.com/stats-js@1.0.1/build/stats.min.js --output third_party/stats.min.js

```


## City-on-Web Workflow


### Training
To reconstruct a large scale scene like _block All_ scene in the _Matrix City_ dataset, simply use:

```
ns-trian block-merf --data path/to/data
```
### Baking 
To bake a reconstructed large scale scene, simply use the following command and load config from the config in the training output folder:
```
ns-trian block-baking --config path/to/config
```

### Real-Time Rendering 


## Citation
```
@article{song2023city,
    title={City-on-Web: Real-time Neural Rendering of Large-scale Scenes on the Web},
    author={Song, Kaiwen and Zhang, Juyong},
    journal={arXiv preprint arXiv:2312.16457},
    year={2023}
    }

```

## Ackownledgements
This repository's code is based on [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) and [MERF](). We are very grateful for their outstanding work.
