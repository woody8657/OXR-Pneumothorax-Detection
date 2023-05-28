# OXR-Pneumothorax-Detection
## Usage
![image](https://github.com/woody8657/OXR-Pneumothorax-Detection/blob/dev/figures/pipeline.png)
Please refer to the folders in `detection`, which contains our implementations of 1) [classification](https://github.com/woody8657/OXR-Pneumothorax-Detection/tree/dev/detection/classification) ,2)  [MMDetection](https://github.com/woody8657/OXR-Pneumothorax-Detection/tree/dev/detection/mmdetection) and 3) [classification](https://github.com/woody8657/OXR-Pneumothorax-Detection/tree/dev/detection/ensemble), as described in my thesis.
## Install MMDetection
1. Create a virtual environment
```
conda create --name pneumothorax_mmdetection python=3.8 -y
```
2. Activate it
```
conda activate pneumothorax_mmdetection
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
8. Uncomment the TensorboardLoggerHook line in configs/_base_/default_runtime.py
## Quick run
### Custom pneumothorax data
Create a symbolink link from /data2
```
mkdir data
ln -s /data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/image/preprocessed/images ./data/images 
```
### Train detector
Example of training faster R-CNN with 8 GPUs. Customize the data, config, models in the config file
```
CUDA_VISIBLE_DEVICES=1,2,3 ./tools/dist_train.sh ./configs_pneumothorax/faster_rcnn.py 3 \
    --work-dir work_dirs/faster_rcnn \
    --auto-scale-lr \
    --seed 42 \
    --deterministic
```
### Evaluate 
Pass the directory that assigned in the training command
```
bash test.sh work_dirs/faster_rcnn
```

## Install YOLOv5~v8
1. Create a virtual environment
```
conda create --name pneumothorax_yolo python=3.8 -y
```
2. Activate it
```
conda activate pneumothorax_yolo
```
3. Install requitements of YOLOv5-v8, the package version is compatible between v58v8. Hence, it's fine to install them in single virtual environment.
```
cd ./detection/yolo_series/YOLOv5/
   ./detection/yolo_series/YOLOv6/
   ./detection/yolo_series/YOLOv7/
   ./detection/yolo_series/YOLOv8/
pip install -r requirements.txt
```