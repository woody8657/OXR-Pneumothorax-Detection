# NTUH20
mkdir ../../reproduce/det-based-cad_retrain/inference_json
bash inf.sh \
    ./configs_pneumothorax_NTUH20/deformable_detr.py \
    /data2/smarted/PXR/data/C426_Pneumothorax_grid/pneumothorax_detection/10_fold/deformable_detr/fold_2 \
    ../../reproduce/det-based-cad_retrain/inference_json/deformable_detr_f2
bash inf.sh \
    ./configs_pneumothorax_NTUH20/deformable_detr.py \
    /data2/smarted/PXR/data/C426_Pneumothorax_grid/pneumothorax_detection/10_fold/deformable_detr/fold_4 \
    ../../reproduce/det-based-cad_retrain/inference_json/deformable_detr_f4
bash inf.sh \
    ./configs_pneumothorax_NTUH20/tood.py \
    /data2/smarted/PXR/data/C426_Pneumothorax_grid/pneumothorax_detection/10_fold/tood/fold_2_positive \
    ../../reproduce/det-based-cad_retrain/inference_json/tood_positive_f2
bash inf.sh \
    ./configs_pneumothorax_NTUH20/tood.py \
    /data2/smarted/PXR/data/C426_Pneumothorax_grid/pneumothorax_detection/10_fold/tood/fold_7 \
    ../../reproduce/det-based-cad_retrain/inference_json/tood_f7
bash inf.sh \
    ./configs_pneumothorax_NTUH20/vfnet.py \
    /data2/smarted/PXR/data/C426_Pneumothorax_grid/pneumothorax_detection/10_fold/vfnet/fold_8_positive \
    ../../reproduce/det-based-cad_retrain/inference_json/vfnet_positive_f8