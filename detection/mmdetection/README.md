# Pneumothorax Detection
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
## Reproducibility
To reproduce the results presented in my thesis, it is necessary to perform 10-fold cross-validation and ensemble different checkpoints.
First of all, [MMDetection](https://github.com/open-mmlab/mmdetection)  follows a config-based design, allowing us to customize any training configuration by modifying a config file.
1. Generate 10-fold config files
```
python make_config.py 
```
The config files will be stored in configs_pneumothorax_10f
2. Training TOOD, VFNet, and Deformable DETR on 10 folds, make sure 3 gpus are available, otherwise the result may be inconsistent.
```
bash train_10f.sh
```
3. Monitoring training logs, training curves, and detailed configurations
```
tensorboard --logdir=/data2/smarted/PXR/data/C426_Pneumothorax_grid/pneumothorax_detection/10_fold/
```
## Inference for ensembling
1. Inference (ex: log_path = /data2/smarted/PXR/data/C426_Pneumothorax_grid/pneumothorax_detection/10_fold/tood/fold_8 outputname = tood_f8)
```
bash test.sh log_path  outputname
```