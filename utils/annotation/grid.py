from pydicom import dcmread
import numpy as np
import cv2
import os
import glob
import pandas as pd
import tqdm

def plot_grid(img, size, width, Mono2, img_max):
    (y,x) = img.shape
    for i in range(size):
        if Mono2:
            color_list = [img_max,img_max,img_max]
            color = tuple([int(xx) for xx in color_list])
        else:
            color = (0,0,0)
        img = cv2.line(img, (int((i+1)*x/size),0), (int((i+1)*x/size),y),color, width) # vertical
        img = cv2.line(img, (0,int((i+1)*y/size)), (x,int((i+1)*y/size)),color, width) # parallel
    return img
def one_patient(dicom, save_path):
    ds = dcmread(dicom)
    os.makedirs(os.path.join(save_path, ds[0x0010,0x0020].value + "_" + ds[0x0008,0x0050].value), exist_ok=True)
    print(ds[0x0010,0x0020].value + "_" + ds[0x0008,0x0050].value)
    # os.makedirs(os.path.join(save_path, ds[0x0008,0x0050].value + "_" + ds[0x0010,0x0020].value), exist_ok=True)
    sop_uid = ds[0x0008,0x0018].value
    series_uid = ds[0x0020,0x000d].value
    for count, grid in enumerate(["none", 10]):
        ds = dcmread(dicom)
        img = ds.pixel_array
        shape = img.shape
        mode  = ds.PhotometricInterpretation
        if grid != "none":
            img = plot_grid(img, grid, 5, mode == "MONOCHROME1", img.max())
         
            # id
            ds[0x0008,0x0018].value = sop_uid + "." + str(count)
            ds[0x0020,0x000e].value = series_uid + "." + str(count)
            try:
                ds[0x0008,0x103e].value = str(grid) + "*" + str(grid)
            except:
                ds.add_new(0x0008103e, 'LO', str(grid) + "*" + str(grid))
            ds[0x0020,0x0011].value = str(count)
        
        ds.PixelData = img
    
        ds[0x0008,0x0018].value =  ds[0x0008,0x0050].value+ "_" + ds[0x0010,0x0020].value
        # accid_pid
        # ds.save_as(os.path.join(save_path, ds[0x0008,0x0050].value+ "_" + ds[0x0010,0x0020].value, str(count) + ".dcm"))
        # pid+_accid
        ds.save_as(os.path.join(save_path, ds[0x0010,0x0020].value+ "_" + ds[0x0008,0x0050].value, str(count) + ".dcm"))
    
    if ds[0x0010,0x0020].value+ "_" + ds[0x0008,0x0050].value in repeat:
        print(ds[0x0010,0x0020].value+ "_" + ds[0x0008,0x0050].value)
    repeat.append(ds[0x0010,0x0020].value+ "_" + ds[0x0008,0x0050].value)




if __name__ == '__main__':
    # save_path = "/home/u/woody8657/data/C426_Pneumothorax_grid/G3_01_dcm/"
    # dcm_path_list = glob.glob(r"/home/u/woody8657/data/C426_Original/C426_Dicom/C426-G3_01/**/**/*.dcm") 
    # df = pd.read_csv("/home/u/woody8657/tmp/C426_G3_01_RADNCLREPORT.csv", encoding= 'unicode_escape')

    save_path = "/home/u/woody8657/data/C426_Pneumothorax_grid/G3_01_dcm"
    dcm_path_list = glob.glob(r'/home/u/woody8657/data/C426_Original/C426_Dicom/C426-G3_01/**/**/*.dcm')
    df = pd.read_csv("/home/u/woody8657/tmp/C426_G4_02_RADNCLREPORT.csv", encoding= 'unicode_escape')
    
    # from multiprocessing import Process, Pool

    # pool = Pool(10)
    # dcm_path_list = [(dcm, save_path) for dcm in dcm_path_list]
   
    # pool.starmap(one_patient, dcm_path_list)

    
    # idx = df[df.iloc[:,1]==0].index.values
    # global repeat
    # one_patient('/home/u/woody8657/data/C426_Original/C426_Dicom/C426_G4/C8-1/002efa13.dcm', save_path)
    # raise
    repeat = []
    not_found = 0
    problem = []
    for dcm in tqdm.tqdm(dcm_path_list):
        

        try:
            one_patient(dcm, save_path)
        except:
            print(print(dcm))
            not_found = not_found+1
            problem.append(dcm)

    print(problem)
    print(f"{not_found} dcm are missing")
    # print(len(dcm_path_list))
    # print(len(os.listdir(save_path)))
# /home/u/woody8657/data/C426_Original/C426_Dicom/C426-G3_01/P214260004857/T0NO027288/FO-5414338680469524534.dcm is missing