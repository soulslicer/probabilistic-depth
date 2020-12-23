# Probabilistic Depth

## About

## Installation

```
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
    python3 train.py --config configs default_mono.json --eval --viz
- Moving Monocular Depth Estimation with Feedback
    python3 train.py --config configs default_mono_feedback.json --eval --viz
- Moving Stereo Depth Estimation
    python3 train.py --config configs default_stereo.json --eval --viz
- Moving Stereo Depth Estimation with Feedback
    python3 train.py --config configs default_stereo_feedback.json --eval --viz
- Moving Monocular Lidar Upsampling
    python3 train.py --config configs default_mono_upsample.json --eval --viz

# Training
    Training only works with Python 3 as it uses distributed training
    To train, simply remove the eval and viz flags. Use the `batch_size` flag to change batch size. It automatically splits it among the GPUs available
    `pkill -f -multi` to clear memory if crashes
```