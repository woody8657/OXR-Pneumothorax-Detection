# OXR-Pneumothorax-Detection
## Install MMDetection
1. Create a vitual environment
```
conda create --name pneumothorax_detection python=3.8 -y
```
2. Activate it
```
conda activate pneumothorax_detection
```
3. Install PyTorch following [official instruction](https://pytorch.org/get-started/locally/). It is important to ensure that the version of CUDA on your system is compatible with PyTorch. For example, CUDA 11.3:
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
4. Install MMCV using MIM. Make sure the version of CUDA a PyTorch you just done.
```
pip install -U openmim
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
```
5. Install the requriments of MMDetection
```
cd detection/mmdetection
pip install -v -e .
```
6. Give yourself permissions to run multi-gpu training
```
chmod +x detection/mmdetection/tools/dist_train.sh
```
7. Install TensorBoard to visualize your training log
```
pip install tensorboardx
```
8. Uncommet the TensorboardLoggerHook line in configs/_base_/default_runtime.py