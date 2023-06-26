# Reproduce
## Reproduce the result in my thesis and paper
1. Activate same virtual environemt with detection's module
```
conda activate pneumothorax_detection
```
2. Install requirements
```
pip install -r requirements.txt
```
3. Classify (stage 1)
```
cd ../detection/classification
conda run -n pneumothorax_classification  bash test.sh
cd ../../reproduce/ 
```
3. Detect (stage 2)
```
cd ../detection/mmdetection/
conda run -n pneumothorax_detection  bash inf_NTUH20.sh
cd ../../reproduce/
```
4. Merge two stage results
```
cd det-based-cad_retrain
conda run -n openmmlab bash ensemble.sh; python 2cls_eliminateion.py 
cd ../
```
5. Run reproduce.sh to reproduce the inference process
```
bash reproduce.sh 
```
If you change the det-based-cad_retrain to det-based-cad in reprodice.sh, the identical numerical with my thesis can be shown by simply use my training result. If you want to reproduce the training process, you can check the following folder. You can find the confifuration and whole training details in there.
```
/data2/smarted/PXR/data/C426_Pneumothorax_grid/reproduce/
```