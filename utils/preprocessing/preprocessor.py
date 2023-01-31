import cv2
import numpy as np
import copy
from scipy.signal import convolve2d  as convolution

class Preprocessor:
    def clahe(self, img, clipLimit=2, tileGridSize=(5,5)):
        img = img.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        img_clahe = clahe.apply(img)
        return img_clahe
    
    def bilateral(self, img):
        img = img.astype(np.uint8)
        return cv2.bilateralFilter(img,18,25,25)
    
    def unsharp_mask(self, img):
        img = img.astype(np.uint8)
        img_denoise = cv2.medianBlur(img,7)
        lap = cv2.Laplacian(img_denoise, ddepth=1,	ksize = 5)
        return img-0.7*lap

    def mycanny(self, img):
        # 1.Noise reduction
        F = np.array([[2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]]) / 159
        output_denoise = convolution(img, F, boundary='symm', mode='same')
        
        # 2.Gradient map and orientation 
        K = 2 # Sobel mask
        # Row gradient
        k_r = np.array([[-1, 0, 1], [-K, 0, K], [-1, 0, 1]])/(K+2)
        row_gdmap = convolution(output_denoise, k_r, boundary='symm', mode='same')
        k_c = np.array([[1, K, 1], [0, 0, 0], [-1, -K, -1]])/(K+2)
        col_gdmap = convolution(output_denoise, k_c, boundary='symm', mode='same')
        output_edge = np.sqrt(np.square(row_gdmap)+np.square(col_gdmap))

        # 3.NMS
        output_NMS = copy.deepcopy(output_edge)
        theta = np.arctan(col_gdmap/row_gdmap)
        for i in range(1,output_edge.shape[0]-1):
            for j in range(1,output_edge.shape[1]-1):
                if theta[i,j] >= 0:
                    if (output_edge[i,j]<=output_edge[i-1,j+1]) or (output_edge[i,j]<=output_edge[i+1,j-1]):
                        output_NMS[i,j] = 0
                else:
                    if (output_edge[i,j]<=output_edge[i+1,j+1]) or (output_edge[i,j]<=output_edge[i-1,j-1]):
                        output_NMS[i,j] = 0

        # 4.Thresholding
        output_thresholding = self._double_thresholding(output_NMS, np.percentile(output_NMS.ravel(), 95), np.percentile(output_NMS.ravel(), 90))
        # 5.Connected
        output_connected = self._connected(output_thresholding)
        
        return output_connected

    def _double_thresholding(self, img, T_H, T_L):
        binary_map = copy.deepcopy(img)
        binary_map[img>T_H] = 255
        binary_map[(img<=T_H) & (img>=T_L)] = 128
        binary_map[img<T_L] = 0
        
        return binary_map 

    def _connected(self, img_mask):
        img = copy.deepcopy(img_mask)
        for i in range(1,img.shape[0]-1):
            for j in range(1,img.shape[1]-1):
                if (img[i,j+1] == 128) and (img[i,j] == 255):
                    img[i,j+1]=255
        for i in range(1,img.shape[0]-1):
            for j in range(1,img.shape[1]-1):
                if (img[img.shape[0]-1-i,img.shape[1]-2-j] == 128) and (img[img.shape[0]-1-i,img.shape[1]-1-j] == 255):
                    img[img.shape[0]-1-i,img.shape[1]-2-j] = 255
        for i in range(1,img.shape[0]-1):
            for j in range(1,img.shape[1]-1):
                if (img[i+1,j] == 128) and (img[i,j] == 255):
                    img[i+1,j]=255
        for i in range(1,img.shape[0]-1):
            for j in range(1,img.shape[1]-1):
                if (img[img.shape[0]-2-i,img.shape[1]-1-j] == 128) and (img[img.shape[0]-1-i,img.shape[1]-1-j] == 255):
                    img[img.shape[0]-2-i,img.shape[1]-1-j] = 255
        img[img==128] = 0
        return img