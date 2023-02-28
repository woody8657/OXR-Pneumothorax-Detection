import os
import glob
import json
from tqdm import tqdm
import numpy as np
import cv2
from matplotlib import pyplot as plt 

def grid_bbox2mask(json_path):
    tmp = json.load(open(json_path))
    mask = np.zeros((tmp['size'][1], tmp['size'][0]))
    for j in range(len(tmp['shapes'])):
        x_a, y_a = tmp['shapes'][j]['a']
        x_b, y_b = tmp['shapes'][j]['b']
        x, y, w, h = int(min(x_a, x_b)), int(min(y_a, y_b)), int(max(x_a, x_b)-min(x_a, x_b)), int(max(y_a, y_b)-min(y_a, y_b))        
        mask[y:y+h,x:x+w] = 1
    
    return mask

if __name__ == '__main__':
    path_list = glob.glob('/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/labels_G301/**/*.json')
    img_path = '/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/images_G301'
    save_path = '/home/u/woody8657/data/C426_Pneumothorax_grid/Pneumothorax_size_threshold_display'
    pixels = []
    thres_33 = 0.0318689602574198
    thres_66 = 0.0804696718241336
    c = 0.8
    s = 0
    m = 0
    l = 0
    size_LUT_G301 = {'all': [], 's':[], 'm':[], 'l':[]}
    for json_path in tqdm(path_list):
        pid = json_path.split('/')[-2]
        img = cv2.imread(os.path.join(img_path, pid+'.png'))
        img[:,:,1] = img[:,:,2] = img[:,:,0]
    
        mask = grid_bbox2mask(json_path)
        num_pixel = len(np.where(mask==1)[0])/(img.shape[0]*img.shape[1])
        pixels.append(num_pixel)
        if num_pixel < thres_33:
            size = 's'
            s = s+1
            size_LUT_G301['s'].append(pid)
            size_LUT_G301['all'].append(pid)
        elif num_pixel >= thres_33 and num_pixel < thres_66:
            size = 'm'
            m = m+1
            size_LUT_G301['m'].append(pid)
            size_LUT_G301['all'].append(pid)
        elif num_pixel >= thres_66:
            size = 'l'
            l = l+1
            size_LUT_G301['l'].append(pid)
            size_LUT_G301['all'].append(pid)
        else:
            raise
        mask = mask.astype('uint8')
        # image = cv2.addWeighted(img, c, np.dstack((mask*0,mask*0,mask*255)), 1-c, 20)
        # cv2.imwrite(os.path.join(save_path, f"{pid}_{size}_{num_pixel}.png"), image)
    print(s,m,l)
    # print(np.percentile(pixels, 33.36))
    # print(np.percentile(pixels, 66.66))
    with open('size_LUT_G301.json', 'w') as file:
        json.dump(size_LUT_G301, file)
    

    # a = np.array(pixels) 
    # plt.hist(a, bins=20) 
    # plt.title("Histogram of relative area") 
    # plt.savefig('hist.png')
        

        