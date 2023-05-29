# Pneumothorax Detection
To reproduce the results presented in my thesis, it is necessary to perform 10-fold cross-validation and ensemble different checkpoints.
First of all, [MMDetection](https://github.com/open-mmlab/mmdetection)  follows a config-based design, allowing us to customize any training configuration by modifying a config file.
1. Generate 10-fold config files
```
python make_config.py 
```
The config files will be stored in configs_pneumothorax_10f
2. Training TOOD, VFNet, and Deformable DETR on 10 folds
```
bash train_10f.sh
```
3. Monitoring training logs, training curves, and detailed configurations
```
tensorboard --logdir=/data2/smarted/PXR/data/C426_Pneumothorax_grid/pneumothorax_detection/10_fold/
```