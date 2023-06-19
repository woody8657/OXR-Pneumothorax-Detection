# Detection Ensemble
## Install requitements
1. Activate it
```
conda activate pneumothorax_mmdetection
```
2. Install requitements 
```
pip install -r requirements.txt
```
## Usage & Example
Enesmmble three models by wbf
```
python ensemble.py \
    --gt ../../data/annotation_coco_format/holdout.json \
    --predictions ../mmdetection/deformable_detr_f3.bbox.json ../mmdetection/tood_f8.bbox.json ../mmdetection/vfnet_f2_p.bbox.json \
    --method wbf  \
    --output-name output.json
```
Evaluate the ensemble result by coco API
```
python cocoeval.py \
    --gt ../../data/annotation_coco_format/holdout.json \
    --prediction output.json
```
Example of Hyperparameter Optimization (HPO) by TPE sampler for searching 1. ensemble algo 2. IoU threshold 3. ensemble weights
```
python ensemble_ray_optuna.py \
    --gt ../../data/annotation_coco_format/holdout.json \
    --jsons ../mmdetection/deformable_detr_f3.bbox.json ../mmdetection/tood_f8.bbox.json ../mmdetection/vfnet_f2_p.bbox.json \
    --output-name output_optuna.json
```