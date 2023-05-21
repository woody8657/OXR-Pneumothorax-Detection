import os

import shutil

import yaml
from yaml.loader import SafeLoader

if __name__ == '__main__':
    for parent in ['images', 'labels']:
        for child in ['train', 'val', 'test']:
            try:
                shutil.rmtree(f'./data/{parent}/{child}')
                os.makedirs(f'./data/{parent}/{child}')
            except:
                os.makedirs(f'./data/{parent}/{child}')
    
    yaml_path = './data/PTX.yaml'
    with open(yaml_path, 'r') as f:
        yaml_dict = yaml.load(f, Loader=SafeLoader)

    for group in ['train', 'val', 'test']:
        with open(yaml_dict[group], 'r') as f:
            img_path_list = f.readlines()
        for img_path in img_path_list:
            pid = img_path.replace('\n', '').split('/')[-1][:-4]
            img_path = img_path.replace('\n','')
            os.symlink(img_path, f'./data/images/{group}/{pid}.png')
            label_path = img_path.replace('images', 'labels').replace('.png','.txt')
            os.symlink(label_path, f'./data/labels/{group}/{pid}.txt')