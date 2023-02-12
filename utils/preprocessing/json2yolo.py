import os
from glob import glob
from tqdm import tqdm
import json
import cv2
import numpy as np
import skimage.measure
import json


class Annotation:
    def __init__(self, label_path, img_prefix, save_dir):
        '''
        File structure:
        |---label_path
            |---pid
                |---label.json
        '''
        self.label_path = label_path
        self.img_prefix = img_prefix
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    def make_label_dir(self, pid_criteria):
        for pid in tqdm(pid_criteria):
            try:
                img = cv2.imread(os.path.join(self.img_prefix, pid+'.png'))
                if pid in pid_criteria:
                    img_w = img.shape[1]
                    img_h = img.shape[0]
                bbox_list = []
                if os.path.isfile(os.path.join(self.label_path, pid, 'label.json')):
                    mask = self.grid_bbox2mask(os.path.join(self.label_path, pid, 'label.json'))
                    xywh_bbox = self.mask2bbox(mask)
                    for x, y, w, h in self.xywh2xcycwh(xywh_bbox):
                        bbox_list.append(f"{0} {x/img_w} {y/img_h} {w/img_w} {h/img_h}\n")
                with open(os.path.join(self.save_dir, pid+'.txt'), 'w') as f:
                    f.writelines(bbox_list)
            except:
                print(pid)
        print(f"{PTX}/{len(pid_criteria)} are positive")
        
        
    def xywh2xcycwh(self, bbox_list):
        return [[x+w/2,y+h/2,w,h] for x,y,w,h in bbox_list]

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
    annotation = Annotation(
            '/home/u/woody8657/data/C426_Pneumothorax_grid/annotation_pid_review/', 
            '/home/u/woody8657/projs/OXR-Pneumothorax-Detection/data/images',
            '/home/u/woody8657/projs/OXR-Pneumothorax-Detection/data/labels'
            )
    pid_list = os.listdir('/home/u/woody8657/data/C426_Pneumothorax_grid/annotation_pid_review/')
    annotation.make_label_dir(pid_list)
        
        
        
        