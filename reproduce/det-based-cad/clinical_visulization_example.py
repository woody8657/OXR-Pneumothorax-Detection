import os
import json
import cv2
from tqdm import tqdm
import numpy as np
import skimage.measure

def grid_bbox2mask(json_path):
    tmp = json.load(open(json_path))
    mask = np.zeros((tmp['size'][1], tmp['size'][0]))
    bbox_list = []
    for j in range(len(tmp['shapes'])):
        x_a, y_a = tmp['shapes'][j]['a']
        x_b, y_b = tmp['shapes'][j]['b']
        x, y, w, h = int(min(x_a, x_b)), int(min(y_a, y_b)), int(max(x_a, x_b)-min(x_a, x_b)), int(max(y_a, y_b)-min(y_a, y_b))        
        mask[y:y+h,x:x+w] = 255
        bbox_list.append([x,y,w,h])
    
    return mask, bbox_list

def mask2bbox(mask):
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

def heatmap_mask(img, mask):
    mask = mask.astype('uint8')
    return cv2.addWeighted(img, c, np.dstack((mask*0,mask*0,mask*255)), 1-c, 20)

class Xray():
    def __init__(self, dicom_path):
        self.dicom = pydicom.read_file(dicom_path)
        self.img_org = self.dicom.pixel_array
        self.img_preprocess, self.uid = read_dcm(dicom_path)
        self.clahe_args = [
            {"clipLimit": 2, "tileGridSize": (5, 5)},
            {"clipLimit": 4., "tileGridSize": (20, 20)}
            ]
    
        self.img_3clahe = clahe_3channel(self.img_preprocess, self.clahe_args)

    def get_img(self):
        mode = self.dicom.PhotometricInterpretation
        img = self.img_org
        if mode == "MONOCHROME1":
            img = np.invert(img)
            img = img - img.min()
        img = np.round_(img / img.max() * 255)
        
        return img.astype(np.float32)

def get_plotted_img(img, bbox, score=None):
    for x,y,w,h in bbox:
        x, y, w, h = int(x), int(y), int(w), int(h)
        img = cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 5)
        if score:
            img = cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 5)
            cv2.putText(img, "Pneumothorax {:.2f}".format(score), (int(x), int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
    return img


class Annotation():
    def __init__(self, json_path):
        self.json = json.load(open(json_path))
        self.bbox = self.get_bbox()
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

if __name__ == '__main__':
    with open('G3_01_mmdet_ver2.json') as file:
        gt = json.load(file)
    with open('ensemble_2cls.json') as file:
        pred = json.load(file)
    
    pid_list = []
    pid_LUT = {}
    for img_meta in gt['images']:
        pid_list.append(img_meta['file_name'][:-4])
        pid_LUT[img_meta['file_name'][:-4]] = img_meta['id']
    gt_p = []
    for annotation_meta in gt['annotations']:
        gt_p.append(gt['images'][annotation_meta['image_id']]['file_name'][:-4])
    gt_n = list(set(pid_list) - set(gt_p))
    gt_p = list(set(gt_p))
    
    pred_p = []
    for annotation_meta in pred:
        if annotation_meta['score'] > 0.20566484218048947:
            pred_p.append(gt['images'][annotation_meta['image_id']]['file_name'][:-4])
    pred_n = list(set(pid_list) - set(pred_p))
    pred_p = list(set(pred_p))

    confusion_matrix = {
        'tp': list(set(gt_p).intersection(set(pred_p))),
        'fp': list(set(gt_n).intersection(set(pred_p))),
        'tn': list(set(gt_n).intersection(set(pred_n))),
        'fn': list(set(gt_p).intersection(set(pred_n)))
    }
    save_path = '/home/u/woody8657/data/C426_Pneumothorax_grid/Clinical_paper_figure2/detection'
    os.makedirs(save_path, exist_ok=True)
    for key in confusion_matrix.keys():
        os.makedirs(os.path.join(save_path, key), exist_ok=True)

    img_path = '/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/images_G301'
    label_path = '/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/labels_G301'
    c = 0.8

    # tp
    case = 'tp'
    for pid in tqdm(confusion_matrix[case]):
        os.makedirs(os.path.join(save_path, case, pid), exist_ok=True)
        img_org = cv2.imread(os.path.join(img_path, pid+'.png'))
        img_org[:,:,2] = img_org[:,:,1] = img_org[:,:,0]
        cv2.imwrite(os.path.join(save_path, case, pid, 'org.png'), img_org)
        mask, _ = grid_bbox2mask(os.path.join(label_path, pid, 'label.json'))
        bbox_list = mask2bbox(mask)
        img_label = heatmap_mask(img_org, mask)
        img_label = get_plotted_img(img_label, bbox_list)
        cv2.imwrite(os.path.join(save_path, case, pid, 'label.png'), img_label)
        img_id = pid_LUT[pid]
        pred_bbox_list = []
        for annotation_meta in pred:
            if annotation_meta['image_id'] == img_id:
                if annotation_meta['score'] > 0.20566484218048947:
                    img_pred = get_plotted_img(img_org, [annotation_meta['bbox']], score=annotation_meta['score'])
        cv2.imwrite(os.path.join(save_path, case, pid, 'pred.png'), img_pred)
        
    
    # fp
    case = 'fp'
    for pid in tqdm(confusion_matrix[case]):
        os.makedirs(os.path.join(save_path, case, pid), exist_ok=True)
        img_org = cv2.imread(os.path.join(img_path, pid+'.png'))
        img_org[:,:,2] = img_org[:,:,1] = img_org[:,:,0]
        cv2.imwrite(os.path.join(save_path, case, pid, 'org.png'), img_org)
        cv2.imwrite(os.path.join(save_path, case, pid, 'label.png'), img_org)
        img_id = pid_LUT[pid]
        pred_bbox_list = []
        for annotation_meta in pred:
            if annotation_meta['image_id'] == img_id:
                if annotation_meta['score'] > 0.20566484218048947:
                    img_pred = get_plotted_img(img_org, [annotation_meta['bbox']], score=annotation_meta['score'])
        cv2.imwrite(os.path.join(save_path, case, pid, 'pred.png'), img_pred)
        
     # fn
    case = 'fn'
    for pid in tqdm(confusion_matrix[case]):
        os.makedirs(os.path.join(save_path, case, pid), exist_ok=True)
        img_org = cv2.imread(os.path.join(img_path, pid+'.png'))
        img_org[:,:,2] = img_org[:,:,1] = img_org[:,:,0]
        cv2.imwrite(os.path.join(save_path, case, pid, 'org.png'), img_org)
        mask, _ = grid_bbox2mask(os.path.join(label_path, pid, 'label.json'))
        bbox_list = mask2bbox(mask)
        img_label = heatmap_mask(img_org, mask)
        img_label = get_plotted_img(img_label, bbox_list)
        cv2.imwrite(os.path.join(save_path, case, pid, 'label.png'), img_label)
        cv2.imwrite(os.path.join(save_path, case, pid, 'pred.png'), img_org)
        
    # tn
    case = 'tn'
    for pid in tqdm(confusion_matrix[case]):
        os.makedirs(os.path.join(save_path, case, pid), exist_ok=True)
        img_org = cv2.imread(os.path.join(img_path, pid+'.png'))
        img_org[:,:,2] = img_org[:,:,1] = img_org[:,:,0]
        cv2.imwrite(os.path.join(save_path, case, pid, 'org.png'), img_org)
        cv2.imwrite(os.path.join(save_path, case, pid, 'label.png'), img_org)
        cv2.imwrite(os.path.join(save_path, case, pid, 'pred.png'), img_org)
        
    
                