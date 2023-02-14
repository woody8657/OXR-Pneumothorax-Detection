import argparse
import os
import sys
# include the preprocessing directory
sys.path.append('../preprocessing')
# importing annotation
from annotation import Annotation

import numpy as np
import cv2

from glob import glob
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann-dir',default="/data2/smarted/PXR/data/C426_Pneumothorax_grid/annotation_pid_review", help='directory of annotations')
    parser.add_argument('--circumscribe',action='store_true', help='circumscribe the grid bounding boxes')
    opt = parser.parse_args()

    ann_path_list = glob(os.path.join(opt.ann_dir, "**", "*.json"))
    
    width_height_ratios = []
    canvas_size = 3000
    canvas = np.zeros((canvas_size,canvas_size,3)) 
    for ann_path in tqdm(ann_path_list):
        annotation = Annotation(ann_path)
        if opt.circumscribe:
            mask, _ = annotation.grid_bbox2mask()
            bbox_list = annotation.mask2bbox(mask)
        else:
            bbox_list = annotation.get_bbox()

        for x, y, w, h in bbox_list:
            x = int(x/annotation.shape[1]*canvas_size)
            y = int(y/annotation.shape[0]*canvas_size)
            w = int(w/annotation.shape[1]*canvas_size)
            h = int(h/annotation.shape[0]*canvas_size)
            canvas = cv2.rectangle(canvas, (x,y), (x+w, y+h), (0, 0, 255), 1)
    
    cv2.imwrite('bbox_dist_circumscribed.png', canvas)