import os
from dicom import DICOM
from preprocessor import Preprocessor

from glob import glob
from tqdm import tqdm

import cv2
import numpy as np

def dcm2preprocessedimg(dcm_path, save_dir):
    
    # cv2.imwrite('tmp.png', dicom.get_img())

    cv2.imwrite('tmp1.png', )

if __name__ == '__main__':
    dcm_path_list = glob('/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/raw/*.dcm')
    save_dir = '/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/images_canny/'
    os.makedirs(save_dir, exist_ok=True)

    for dcm_path in tqdm(dcm_path_list):
        dicom = DICOM(dcm_path)
        preprocessor = Preprocessor()
        pid = dicom.get_metadata()[0x0010, 0x0020].value
        preprocessed_img = preprocessor.mycanny(dicom.get_img()).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, pid+'.png'), cv2.cvtColor(preprocessed_img,cv2.COLOR_GRAY2BGR))