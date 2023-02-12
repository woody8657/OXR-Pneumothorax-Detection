import os
import json

if __name__ == '__main__':
    img_prefix = "/home/u/woody8657/projs/OXR-Pneumothorax-Detection/data/images"
    json_path = "/home/u/woody8657/projs/OXR-Pneumothorax-Detection/data/annotation_coco_format/holdout.json"
    save_txt_path = "/home/u/woody8657/projs/OXR-Pneumothorax-Detection/data/yolo_group/holdout.txt"

    with open(json_path, 'r') as f:
        ann_dict = json.load(f)
    image_path_list = []
    for image in ann_dict['images']:
        image_path_list.append(os.path.join(img_prefix, image['file_name'])+'\n')
    with open(save_txt_path, 'w') as f:
        f.writelines(image_path_list)
