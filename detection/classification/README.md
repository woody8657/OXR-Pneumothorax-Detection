# OXR-Pneumothorax-Detection
## Classification stage
The training code is integrated with pytorch lightning.
1. 
```
pip install -r requirements.txt
```
2. 
Check the example in the train.sh and test.sh. You can custom the training config via the json files in configs/. The deault configs in configs/ input 1024*1024 image size, festivo is recommended and make sure 2 gpus are available, otherwise the result may be inconsistent.
```
bash train.sh
```
3. Modify the logs path in ```test.py``` then we can ensemble different models.
```
bash test.sh
```