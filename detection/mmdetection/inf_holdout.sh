# NTUH1519 holdout
mkdir ../ensemble/inference_json_holdout
bash test.sh \
    /data2/smarted/PXR/data/C426_Pneumothorax_grid/pneumothorax_detection/10_fold/deformable_detr/fold_2 \
    ../ensemble/inference_json_holdout/deformable_detr_f2
bash test.sh \
    /data2/smarted/PXR/data/C426_Pneumothorax_grid/pneumothorax_detection/10_fold/deformable_detr/fold_4 \
    ../ensemble/inference_json_holdout/deformable_detr_f4
bash test.sh \
    /data2/smarted/PXR/data/C426_Pneumothorax_grid/pneumothorax_detection/10_fold/tood/fold_2_positive \
    ../ensemble/inference_json_holdout/tood_positive_f2
bash test.sh \
    /data2/smarted/PXR/data/C426_Pneumothorax_grid/pneumothorax_detection/10_fold/tood/fold_7 \
    ../ensemble/inference_json_holdout/tood_f7
bash test.sh \
    /data2/smarted/PXR/data/C426_Pneumothorax_grid/pneumothorax_detection/10_fold/vfnet/fold_8_positive \
    ../ensemble/inference_json_holdout/vfnet_positive_f8