import json
import glob
import pandas as pd
import numpy as np
import cv2
import pydicom
import os
import math
save_dir = "/home/u/woody8657/data/C426_Pneumothorax_grid/IOU_threshold_comparison_301~645＿missing16/"
fdir = os.makedirs(save_dir, exist_ok=True)
for i in range(10):
    fdir = os.makedirs(save_dir + 'IOU:'+str(round(i*0.1,2))+"~"+str(round((i+1)*0.1,2)) +'/', exist_ok=True)


missing = ['T0NO004088', 'T0NO004354', 'T0NO004503', 'T0NO004617', 'T0NO004726', 'T0NO005474', 'T0NO006593', 'T0NO007193', 'T0NO009931', 'T0NO011479', 'T0NO012865', 'T0NO015061', 'T0NO015523', 'T0NO022271', 'T0NO023000', 'T0NO023809']

img_path = "/data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/image/raw/"
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

df = pd.read_csv("./C426_G1_01_RADNCLREPORT.csv")

ann_path = glob.glob(r"/home/u/woody8657/data/C426_Pneumothorax_grid/annotation_4/**/*.json")
acc_list = []
repeat_list = []
for i in ann_path:
    tmp = json.load(open(i))
    acc_list.append(tmp['accessionNumber'])
dict = {}
for key in acc_list:
    dict[key] = dict.get(key, 0) + 1
for key in dict.keys():
    if dict[key]==1:
        repeat_list.append(key)
# print(dict)
print(repeat_list)
IOU_list = []


for count, accession in enumerate(repeat_list):
    img_list = []
    mask_list = []
    print(f'processing {count}th image...')
    for i in ann_path:
      
        tmp = json.load(open(i))
        
        if tmp['accessionNumber'] == accession:
            
            idx = df[df['ACCESSNO2']==tmp['accessionNumber']].index.values[0]
            patient_id = df.iloc[idx,0]
            file_name = patient_id + ".png"
            ds = pydicom.read_file(img_path + patient_id + ".dcm") # read dicom image
            mode  = ds.PhotometricInterpretation
            img = ds.pixel_array # get image array
            if mode == "MONOCHROME1":
                img = np.invert(img)
                img = img - img.min()
            img = np.round_(img / img.max() * 255)
            img = np.float32(img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img_org = np.copy(img)
            mask = np.zeros((tmp['size'][1],tmp['size'][0]))
            for j in range(len(tmp['shapes'])):
                x_a, y_a = tmp['shapes'][j]['a']
                x_b, y_b = tmp['shapes'][j]['b']
                (x, y, w, h) = map(int,(round(min(x_a, x_b)), round(min(y_a, y_b)), round(max(x_a, x_b))-round(min(x_a, x_b)), round(max(y_a, y_b)-min(y_a, y_b))))
                img = cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 10)
                x, y, w, h = round(min(x_a, x_b)), round(min(y_a, y_b)), round(max(x_a, x_b))-round(min(x_a, x_b)), round(max(y_a, y_b)-min(y_a, y_b))
                mask[y:y+h,x:x+w] = 255
            img_list.append(img)
            mask_list.append(mask)
    try:
        mask_tmp = mask_list[0] + mask_list[1]
        IOU = idx1 = len(np.argwhere(mask_tmp >= 510))/len(np.argwhere(mask_tmp >= 255))
    except:
        IOU = 0
    IOU_list.append(IOU)
    # print(math.floor(IOU * 10) / 10.0)
    # print(len(img_list))
    # print(save_dir + 'IOU:'+ str(math.floor(IOU * 10) / 10.0)  + '~' + str(math.floor(IOU * 10) / 10.0+0.1)  + '/' + patient_id + '/')
    # fdir = os.makedirs(save_dir + 'IOU:'+ str(math.floor(IOU * 10) / 10.0)  + '~' + str(math.floor(IOU * 10) / 10.0+0.1)  + '/' + accession + '_'+ patient_id + "_IOU:"  + str(round(IOU,2))+ '/', exist_ok=True)
    # cv2.imwrite(save_dir + 'IOU:'+ str(math.floor(IOU * 10) / 10.0)  + '~' + str(math.floor(IOU * 10) / 10.0+0.1)  + '/' + accession + '_'+ patient_id + "_IOU:"  + str(round(IOU,2)) + '/' + ' org.png', img_org)
    # cv2.imwrite(save_dir + 'IOU:'+ str(math.floor(IOU * 10) / 10.0)  + '~' + str(math.floor(IOU * 10) / 10.0+0.1)  + '/' + accession + '_'+ patient_id + "_IOU:"  + str(round(IOU,2)) + '/' + ' annotation0.png', img_list[0])
    # cv2.imwrite(save_dir + 'IOU:'+ str(math.floor(IOU * 10) / 10.0)  + '~' + str(math.floor(IOU * 10) / 10.0+0.1)  + '/' + accession + '_' +patient_id + "_IOU:"  + str(round(IOU,2)) + '/' + ' annotation1.png', img_list[1])
    # cv2.imwrite("/data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/image/segmentation/mask/" + file_name, mask)
    fdir = os.makedirs(save_dir + '/' + accession + '_'+ patient_id +'/', exist_ok=True)
    cv2.imwrite(save_dir + accession + '_'+ patient_id + '/' + ' org.png', img_org)
    cv2.imwrite(save_dir + accession + '_'+ patient_id + '/' + ' annotation0.png', img_list[0])
    
print(np.mean(np.array(IOU_list)))
print(np.std(np.array(IOU_list)))


import matplotlib.pyplot as plt



n, bins, patches=plt.hist(IOU_list)
plt.xlabel("IOU")
plt.ylabel("Counts")
plt.title("IOU of different doctors' annotations")
plt.savefig("hist.png")