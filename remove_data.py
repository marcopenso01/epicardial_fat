import os
import numpy as np
import h5py
import cv2
import pydicom # for reading dicom files
import matplotlib.pyplot as plt

def crop_or_pad_slice_to_size(slice, nx, ny):
    
    if len(slice.shape) == 3:
        stack = [slice[:,:,0], slice[:,:,1], slice[:,:,2]]
        RGB = []
    else:
        stack = [slice]
    
    for i in range(len(stack)):
        
        img = stack[i]
            
        x, y = img.shape
        
        x_s = (x - nx) // 2
        y_s = (y - ny) // 2
        x_c = (nx - x) // 2
        y_c = (ny - y) // 2
    
        if x > nx and y > ny:
            slice_cropped = img[x_s:x_s + nx, y_s:y_s + ny]
        else:
            slice_cropped = np.zeros((nx, ny))
            if x <= nx and y > ny:
                slice_cropped[x_c:x_c + x, :] = img[:, y_s:y_s + ny]
            elif x > nx and y <= ny:
                slice_cropped[:, y_c:y_c + y] = img[x_s:x_s + nx, :]
            else:
                slice_cropped[x_c:x_c + x, y_c:y_c + y] = img[:, :]
        if len(stack)>1:
            RGB.append(slice_cropped)
    
    if len(stack)>1:
        return np.dstack((RGB[0], RGB[1], RGB[2]))
    else:
        return slice_cropped
    
    
path = ''


for paz in os.listdir(path):
    print(paz)
    LIST = []
    fold_paz = os.path.join(path, paz)

    path_seg = os.path.join(fold_paz, 'seg')
    path_seg = os.path.join(path_seg, os.listdir(path_seg)[0])
    path_raw = os.path.join(fold_paz, 'raw')
    path_raw = os.path.join(path_raw, os.listdir(path_raw)[0])

    for i in range(len(os.listdir(path_seg))):
        dcmPath = os.path.join(path_seg, os.listdir(path_seg)[i])
        data_row_img = pydicom.dcmread(dcmPath)
        img = data_row_img.pixel_array
        img = crop_or_pad_slice_to_size(img, 310, 310)
        flag = 1
        for r in range(0, img.shape[0]):
            for c in range(0, img.shape[1]):
                if not img[r,c,0] == img[r,c,1] == img[r,c,2]:
                    flag = 0
        if flag == 1:
            LIST.append(os.path.join(path_seg, os.listdir(path_seg)[i]))
            LIST.append(os.path.join(path_raw, os.listdir(path_raw)[i]))

    for i in range(len(LIST)):
        os.remove(LIST[i])
