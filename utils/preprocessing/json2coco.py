import os
from glob import glob
from tqdm import tqdm
import json
import cv2
import numpy as np
import skimage.measure
import json


class Annotation:
    def __init__(self, label_path, img_prefix):
        '''
        File structure:
        |---label_path
            |---pid
                |---label.json
        '''
        self.label_path = label_path
        self.img_prefix = img_prefix

    def make_images_list(self, pid_criteria):
        print('making images list...')
        self.image_id_LUT = {}
        image_list = []
        id = 0
        for pid in tqdm(pid_criteria):
            img = cv2.imread(os.path.join(self.img_prefix, pid+'.png'))
            if pid in pid_criteria:
                image_list.append({
                    "id": id,
                    "width": img.shape[1],
                    "height": img.shape[0],
                    "file_name": pid + '.png',
                })
                self.image_id_LUT[pid] = id
                id = id + 1
        return image_list
       
    
    def make_annotations_list(self, pid_criteria):
        print('making annotations list...')
        anntation_list = []
        id = 0
        for pid in tqdm(pid_criteria):
            if os.path.isfile(os.path.join(self.label_path, pid, 'label.json')):
                mask = self.grid_bbox2mask(os.path.join(self.label_path, pid, 'label.json'))
                xywh_bbox = self.mask2bbox(mask)
                for x, y, w, h in xywh_bbox:
                    anntation_list.append({
                        "id": id,
                        "image_id": self.image_id_LUT[pid],
                        "category_id": 0,
                        "segmentation": [],
                        "area": w*h,
                        "bbox": [x,y,w,h],
                        "iscrowd": 0,
                    })
                    id = id + 1
        return anntation_list

    def make_categories_list(self, finding):
        categories = [{
            "id": 0,
            "name": finding,
        }]
        return categories

    def make_json(self, criteria, json_name):
        image_list = self.make_images_list(criteria)
        annotation_list = self.make_annotations_list(criteria)
        categories = self.make_categories_list('Pneumothorax')
        coco_format_json = dict(
        images=image_list,
        annotations=annotation_list,
        categories=categories)

        with open(json_name, 'w') as f:
            json.dump(coco_format_json, f)

    def grid_bbox2mask(self, json_path):
        tmp = json.load(open(json_path))
        mask = np.zeros((tmp['size'][1], tmp['size'][0]))
        for j in range(len(tmp['shapes'])):
            x_a, y_a = tmp['shapes'][j]['a']
            x_b, y_b = tmp['shapes'][j]['b']
            x, y, w, h = int(min(x_a, x_b)), int(min(y_a, y_b)), int(max(x_a, x_b)-min(x_a, x_b)), int(max(y_a, y_b)-min(y_a, y_b))        
            mask[y:y+h,x:x+w] = 255
        
        return mask

    def mask2bbox(self, mask):
        mask[mask<125]=0
        mask[mask>=125]=1
        mask = skimage.measure.label(mask,connectivity=1)
        xyxy_bbox = np.empty((0,4), int)
        
        for region in skimage.measure.regionprops(mask):
            minr, minc, maxr, maxc = region.bbox
            xyxy_bbox = np.append(xyxy_bbox, np.expand_dims(np.array([minc, minr, maxc, maxr]), axis=0), axis=0)
    
        xywh_bbox = []
        for i in range(xyxy_bbox.shape[0]):
            (x1,y1) = map(int,(xyxy_bbox[i,0], xyxy_bbox[i,1]))
            (x2,y2) = map(int,(xyxy_bbox[i,2], xyxy_bbox[i,3]))
            xywh_bbox.append([x1, y1, x2-x1, y2-y1])
            
        return xywh_bbox



if __name__ == '__main__':
    annotation = Annotation('/home/u/woody8657/tmp/Annotation_cleaning/G3_01_annotation_pid_review/', 
            '/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/images_G301')
    pid_list = os.listdir('/home/u/woody8657/tmp/Annotation_cleaning/G3_01_annotation_pid_review')

    annotation.make_json(pid_list, '.json')
        
        
        
        