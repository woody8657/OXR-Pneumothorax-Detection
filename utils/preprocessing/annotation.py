import os
import json
import numpy as np
import cv2
import skimage.measure

class Annotation():
    def __init__(self, json_path):
        self.json = json.load(open(json_path))
        self.accessionNumber = self.json['accessionNumber']
        self.shape = self.json['size'][::-1]

    def get_bbox(self):
        bbox_list = []
        for j in range(len(self.json['shapes'])):
            x_a, y_a = self.json['shapes'][j]['a']
            x_b, y_b = self.json['shapes'][j]['b']
            # left up corner
            (x, y, w, h) = map(int,(round(min(x_a, x_b)), round(min(y_a, y_b)), round(max(x_a, x_b))-round(min(x_a, x_b)), round(max(y_a, y_b)-min(y_a, y_b))))
            x, y, w, h = round(min(x_a, x_b)), round(min(y_a, y_b)), round(max(x_a, x_b))-round(min(x_a, x_b)), round(max(y_a, y_b)-min(y_a, y_b))
            bbox_list.append([x,y,w,h])
        
        return bbox_list

    def get_plotted_img(self, img, bbox):
        # img = self.get_img()

        for x,y,w,h in bbox:
            img = cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 5)
            # score = np.random.rand() / 10 + 0.9
            # cv2.putText(img, "Pneumothorax {:.2f}".format(score), (int(x), int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        return img
    

    def grid_bbox2mask(self):
        mask = np.zeros((self.json['size'][1], self.json['size'][0]))
        bbox_list = []
        for j in range(len(self.json['shapes'])):
            x_a, y_a = self.json['shapes'][j]['a']
            x_b, y_b = self.json['shapes'][j]['b']
            x, y, w, h = int(min(x_a, x_b)), int(min(y_a, y_b)), int(max(x_a, x_b)-min(x_a, x_b)), int(max(y_a, y_b)-min(y_a, y_b))        
            mask[y:y+h,x:x+w] = 255
            bbox_list.append([x,y,w,h])
        
        return mask, bbox_list

    def mask2bbox(self, mask):
        mask[mask<125]=0
        mask[mask>=125]=1
        tmp_mask = np.zeros((2500, 2048, 3))
        mask = skimage.measure.label(mask,connectivity=1)
        xyxy_bbox = np.empty((0,4), int)
        
        for count, region in enumerate(skimage.measure.regionprops(mask)):
            minr, minc, maxr, maxc = region.bbox
            xyxy_bbox = np.append(xyxy_bbox, np.expand_dims(np.array([minc, minr, maxc, maxr]), axis=0), axis=0)
            if count == 0:
                tmp_mask[minr:maxr, minc:maxc,:] = np.array([0,0,240])
            if count == 1:
                tmp_mask[minr:maxr, minc:maxc,:] = np.array([0,0,120])

        xywh_bbox = []
        for i in range(xyxy_bbox.shape[0]):
            (x1,y1) = map(int,(xyxy_bbox[i,0], xyxy_bbox[i,1]))
            (x2,y2) = map(int,(xyxy_bbox[i,2], xyxy_bbox[i,3]))
            xywh_bbox.append([x1, y1, x2-x1, y2-y1])
            
        return xywh_bbox