cd det-based-cad

# mAP
python cocoeval.py --gt ../../data/10f_traininglist/holdout.json --pred ./prediction/ensemble_NTUH_1519.json
python cocoeval.py --gt ./prediction/NTUH_20_gt.json --pred ./prediction/ensemble.json

# classification
for size in overall l m s
do
    echo Diagnostic performance of pneumothorax size: $size
    python diagnose_cad.py --bbox-size $size
    python diagnose_radiologist.py --bbox-size $size
done

# localization
echo Localizatioin performance of pneumothorax:
python localize.py
python localize_annotator.py.py 
