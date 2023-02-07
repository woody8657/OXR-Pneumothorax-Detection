import os
from multiprocessing import Pool

from dicom import DICOM

from preprocessor import Preprocessor

import cv2
import numpy as np

from glob import glob
from tqdm import tqdm

def preprocess(dcm_path):
    dicom = DICOM(dcm_path)
    preprocessor = Preprocessor()
    pid = dicom.get_metadata()[0x0010, 0x0020].value
    preprocessed_img = dicom.get_img()
    # preprocessed_img = preprocessor.mycanny(dicom.get_img()).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, pid+'.png'), cv2.cvtColor(preprocessed_img.astype(np.uint8),cv2.COLOR_GRAY2BGR))

if __name__ == '__main__':
    dcm_path_list = glob('/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/raw/*.dcm')
    global save_dir
    save_dir = '/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/images_org/'
    os.makedirs(save_dir, exist_ok=True)

    with Pool(30) as p:
        p.map(preprocess, dcm_path_list)
            