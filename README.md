# Probabilistic Depth

## About

This code predicts depth from RGB images. But instead of producing depth alone, it produces a multimodal depth distribution for each pixel, in the form of a categorical distribution. This is useful for weeding out uncertain 3d points, or in downstream adaptive depth sensing tasks. We solve the task of Monocular, Stereo and Lidar Upsampling based depth estimation using the same architecture.

Monocular
![Monocular](https://github.com/soulslicer/probabilistic-depth/blob/main/pics/mono.gif?raw=true)

Stereo
![Stereo](https://github.com/soulslicer/probabilistic-depth/blob/main/pics/stereo.gif?raw=true)

Upsampling
![Upsampling](https://github.com/soulslicer/probabilistic-depth/blob/main/pics/upsample.gif?raw=true)

![Upsampling](https://github.com/soulslicer/probabilistic-depth/blob/main/pics/ptcloud_stereo.gif?raw=true)

## Explanation

Neural RGBD introduced the world to Depth Probability Fields (DPV). We build off that idea and produce an easy to understand and extend code. Instead of predicting depth per pixel, we predict a distribution per pixel. To help us visualize the uncertainty, we collapsed the distribution along the surface of the road so that you can visualize the Uncertainty Field (UF). You can read more details in [here](https://github.com/soulslicer/probabilistic-depth/blob/main/pics/explanation.pdf) 

<img src="https://raw.githubusercontent.com/soulslicer/probabilistic-depth/main/pics/image1.png" width="500" height="300" />

## Installation

```
- Download Kitti
    # Download from http://www.cvlibs.net/datasets/kitti/raw_data.php
    # Use the raw_kitti_downloader script
    # Link kitti to a folder in this project folder
    ln -s {absolute_kitti_path} kitti
- Install Python Deps
    torch, matplotlib, numpy, cv2, pykitti, tensorboardX
- Compile external deps
    cd external
    # Select the correct sh file depending on your python version
    # Modify it so that you select the correct libEGL version
    # Eg. /usr/lib/x86_64-linux-gnu/libEGL.so.1.1.0
    sudo sh compile_3.X.sh
    # Ensure everything compiled correctly with no errors
- Download pre-trained models
    # https://drive.google.com/file/d/1XN5lrFobInkcJ4F6cHgAd385eX9Galfw/view?usp=sharing
    unzip output.zip
```

## Running

```
# Eval
- Moving Monocular Depth Estimation
    python3 train.py --config configs/default_mono.json --eval --viz
- Moving Monocular Depth Estimation with Feedback
    python3 train.py --config configs/default_mono_feedback.json --eval --viz
- Stereo Depth Estimation
    python3 train.py --config configs/default_stereo.json --eval --viz
- Stereo Depth Estimation with Feedback
    python3 train.py --config configs/default_stereo_feedback.json --eval --viz
- Moving Monocular Lidar Upsampling
    python3 train.py --config configs/default_mono_upsample.json --eval --viz

# Training
    Training only works with Python 3 as it uses distributed training
    To train, simply remove the eval and viz flags. Use the `batch_size` flag to change batch size. It automatically splits it among the GPUs available
    `pkill -f -multi` to clear memory if crashes
```

## Math?

Coming soon. Disclaimer: This is a template code that I have written to extend some research I am working on. Please be sure to cite this if you use this code

## References

```
Uses code from https://github.com/lliuz/ARFlow (ARFlow) and https://github.com/NVlabs/neuralrgbd (NeuralRGBD)
```