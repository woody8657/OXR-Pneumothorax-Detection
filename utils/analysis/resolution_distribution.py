import argparse
import os

import pydicom
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dcm-dir',default="/data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/image/raw", help='directory of DICOMs')
    opt = parser.parse_args()

    dcm_path_list = glob(os.path.join(opt.dcm_dir, "*.dcm"))
    
    height = []
    width = []
    for dcm_path in tqdm(dcm_path_list):
        ds = pydicom.dcmread(dcm_path)
        height.append(ds[0x00280010].value)
        width.append(ds[0x00280011].value)

    print(f"Image height mean: {np.mean(height)}, std: {np.std(height)}.")
    plt.hist(height)
    plt.title("Histogram of images' height")
    plt.savefig("height_dist.png")
    print(f"Image width mean: {np.mean(width)}, std: {np.std(width)}.")
    plt.clf()
    plt.hist(width)
    plt.title("Histogram of images' width")
    plt.savefig("width_dist.png")

