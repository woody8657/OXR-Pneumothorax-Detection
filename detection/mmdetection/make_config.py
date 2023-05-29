import os
from config_template_10f import get_config_text

if __name__ == '__main__':
    save_path = 'configs_pneumothorax_10f'
    os.makedirs(save_path, exist_ok=True)
    for fold in range(1,11):
        for model in ['tood', 'deformable_detr', 'vfnet']:
            for train_positive in [True, False]:
                suffix = ''
                if train_positive:
                    suffix = '_positive'
                with open(os.path.join(save_path, f'{model}_f{fold}{suffix}.py'), 'w') as file:
                    text = get_config_text(model, fold, '/data2/smarted/PXR/data/C426_Pneumothorax_grid/pneumothorax_detection/10_fold', train_positive=train_positive)
                    file.writelines(text)
