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
## Usage
Example
```
python ensemble.py \
    --gt ground_truth.json \
    --predictions prediction1.json prediction2.json prediction3.json ... \
    --method wbf  \
    --output-name output.json
```