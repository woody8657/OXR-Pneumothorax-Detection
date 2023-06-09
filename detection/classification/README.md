# Pneumothorax Classification
## Install requitements
1. Create a virtual environment
```
conda create --name pneumothorax_classification python=3.8 -y
```
2. Activate it
```
conda activate pneumothorax_classification
```
3. Install requitements 
```
pip install -r requirements.txt
```
## Reproducibility
The training code is integrated with pytorch lightning.
1. Check the example in the train.sh and test.sh. You can custom the training config via the json files in configs/. The deault configs in configs/ input 1024*1024 image size, festivo is recommended and make sure 2 gpus are available, otherwise the result may be inconsistent.
```
bash train.sh
```
2. Modify the logs path in ```test.py``` then we can ensemble different models.
```
bash test.sh
```