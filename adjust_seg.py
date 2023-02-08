# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 10:45:05 2023

@author: Marco Penso
"""

import scipy
import scipy.io
import os
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
import skimage.morphology, skimage.data
import math
import random
import pydicom
from skimage import color
X = []
Y = []

drawing=False # true if mouse is pressed
mode=True

def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False


def paint_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
                cv2.line(image_binary,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
                current_former_x = former_x
                current_former_y = former_y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
            cv2.line(image_binary,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
            current_former_x = former_x
            current_former_y = former_y
    return former_x,former_y
  
def imfill(img, dim):
    img = img[:,:,0]
    img = cv2.resize(img, (dim, dim))
    img[img>0]=255
    im_floodfill = img.copy()
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    return img | cv2.bitwise_not(im_floodfill)


# path
input_path = r'F:/BED-REST/patientsESA20221202/patientsESA20221202/pre_process/paz10'
phase = 17
# read data
data = h5py.File(os.path.join(input_path, 'pred.hdf5'), 'r')

MASK = []
RAW = []

for i in range(len(data['img_raw'])):
    
    if data['phase'][i] == str(phase):
        
    
        print('-------------------')
        print('%d/%d' % (i+1, len(data['img_raw'])))
        img_raw = cv2.normalize(src=data['img_raw'][i], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                dtype=cv2.CV_8U)
        dim = img_raw.shape[0]
        pred = data['pred'][i].astype(np.uint8)
        fig = plt.figure(figsize=(14, 14))
        ax1 = fig.add_subplot(121)
        ax1.set_axis_off()
        ax1.imshow(img_raw, cmap='gray')
        ax2 = fig.add_subplot(122)
        ax2.set_axis_off()
        ax2.imshow(
            color.label2rgb(pred, img_raw, colors=[(255, 0, 0), (0, 255, 0), (255, 255, 0)], alpha=0.0008, bg_label=0,
                            bg_color=None))
        ax1.title.set_text('Raw_img')
        ax2.title.set_text('Automated')
        plt.show()
        flag=True
        while flag:
            print('1: add seg; 2: canc seg; 3: remove image; 4: ok, 5: remove seg')
            c = input("Enter a command: ")
            print(c)
            if c == '1' or c == '2' or c == '3' or c == '4' or c == '5':
                flag=False
               
        if c == '2':
            print(' --- canc seg ---')
            mask = pred.copy()
            mask[mask != 1] = 0        
            tit=['---Segmenting Fat---']
            m = mask.copy()
            m = cv2.normalize(src=m, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            img = data['img_raw'][i].copy()
            img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            img = cv2.drawContours(img, contours, -1, (255, 255, 255), 1)
            img = cv2.resize(img, (400, 400), interpolation = cv2.INTER_CUBIC)
            image_binary = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
            cv2.namedWindow(tit[0])
            cv2.setMouseCallback(tit[0],paint_draw)
            while(1):
                cv2.imshow(tit[0],img)
                k=cv2.waitKey(1)& 0xFF
                if k==27: #Escape KEY
                    im_out = imfill(image_binary, dim)
                    break              
            cv2.destroyAllWindows()
            im_out[im_out>0]=1
            coord = np.where(im_out>0)
            for nn in range(len(coord[0])):
                mask[coord[0][nn],coord[1][nn]]=0
            # plot data
            fig = plt.figure(figsize=(14, 14))
            ax1 = fig.add_subplot(121)
            ax1.set_axis_off()
            ax1.imshow(img_raw, cmap='gray')
            ax2 = fig.add_subplot(122)
            ax2.set_axis_off()
            ax2.imshow(
                color.label2rgb(mask.astype(np.uint8), img_raw, colors=[(255, 0, 0), (0, 255, 0), (255, 255, 0)], alpha=0.0008, bg_label=0,
                                bg_color=None))
            ax1.title.set_text('Raw_img')
            ax2.title.set_text('Automated')
            plt.show()
            mask = mask.astype(np.uint8)
            
        if c == '3':
            print(' --- remove image ---')
            continue
        
        if c == '4':
            print(' --- ok ---')
            MASK.append(data['pred'][i].astype(np.uint8))
            RAW.append(data['img_raw'][i])
            
        if c == '5':
            print(' --- remove seg ---')
            RAW.append(data['img_raw'][i])
            MASK.append(np.zeros((dim,dim), dtype=np.uint8))
            
        if c == '1':
            print(' --- add seg ---')
            mask = pred.copy()
            mask[mask != 1] = 0        
            tit=['---Segmenting Fat---']
            m = mask.copy()
            m = cv2.normalize(src=m, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            img = data['img_raw'][i].copy()
            img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            img = cv2.drawContours(img, contours, -1, (255, 255, 255), 1)
            img = cv2.resize(img, (400, 400), interpolation = cv2.INTER_CUBIC)
            image_binary = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
            cv2.namedWindow(tit[0])
            cv2.setMouseCallback(tit[0],paint_draw)
            while(1):
                cv2.imshow(tit[0],img)
                k=cv2.waitKey(1)& 0xFF
                if k==27: #Escape KEY
                    im_out = imfill(image_binary, dim)
                    break              
            cv2.destroyAllWindows()
            im_out[im_out>0]=1
            im_out = im_out + mask
            im_out[im_out>0]=1
            # plot data
            fig = plt.figure(figsize=(14, 14))
            ax1 = fig.add_subplot(121)
            ax1.set_axis_off()
            ax1.imshow(img_raw, cmap='gray')
            ax2 = fig.add_subplot(122)
            ax2.set_axis_off()
            ax2.imshow(
                color.label2rgb(im_out.astype(np.uint8), img_raw, colors=[(255, 0, 0), (0, 255, 0), (255, 255, 0)], alpha=0.0008, bg_label=0,
                                bg_color=None))
            ax1.title.set_text('Raw_img')
            ax2.title.set_text('Automated')
            plt.show()
            mask = im_out.astype(np.uint8)
            
        
        if c == '1' or c == '2':
            while(1):      
                print('1: add seg; 2: canc seg; 0: esc')
                c = input("Enter a command: ")
                print(c)
                
                if c=='0':
                    print('--- esc ---')
                    RAW.append(data['img_raw'][i])
                    MASK.append(mask)
                    break
                
                if c=='1':
                    print(' --- add seg ---')
                    
                    dim = img_raw.shape[0]            
                    tit=['---Segmenting Fat---']
                    m = mask.copy()
                    m = cv2.normalize(src=m, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    
                    img = data['img_raw'][i].copy()
                    img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    img = cv2.drawContours(img, contours, -1, (255, 255, 255), 1)
                    img = cv2.resize(img, (400, 400), interpolation = cv2.INTER_CUBIC)
                    image_binary = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
                    cv2.namedWindow(tit[0])
                    cv2.setMouseCallback(tit[0],paint_draw)
                    while(1):
                        cv2.imshow(tit[0],img)
                        k=cv2.waitKey(1)& 0xFF
                        if k==27: #Escape KEY
                            im_out = imfill(image_binary, dim)
                            break              
                    cv2.destroyAllWindows()
                    im_out[im_out>0]=1
                    im_out = im_out + mask
                    im_out[im_out>0]=1
                    # plot data
                    fig = plt.figure(figsize=(14, 14))
                    ax1 = fig.add_subplot(121)
                    ax1.set_axis_off()
                    ax1.imshow(img_raw, cmap='gray')
                    ax2 = fig.add_subplot(122)
                    ax2.set_axis_off()
                    ax2.imshow(
                        color.label2rgb(im_out.astype(np.uint8), img_raw, colors=[(255, 0, 0), (0, 255, 0), (255, 255, 0)], alpha=0.0008, bg_label=0,
                                        bg_color=None))
                    ax1.title.set_text('Raw_img')
                    ax2.title.set_text('Automated')
                    plt.show()
                    mask = im_out.astype(np.uint8)
                    
                if c == '2':
                    print(' --- canc seg ---')
                    dim = img_raw.shape[0]
                    
                    tit=['---Segmenting Fat---']
                    m = mask.copy()
                    m = cv2.normalize(src=m, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    
                    img = data['img_raw'][i].copy()
                    img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    img = cv2.drawContours(img, contours, -1, (255, 255, 255), 1)
                    img = cv2.resize(img, (400, 400), interpolation = cv2.INTER_CUBIC)
                    image_binary = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
                    cv2.namedWindow(tit[0])
                    cv2.setMouseCallback(tit[0],paint_draw)
                    while(1):
                        cv2.imshow(tit[0],img)
                        k=cv2.waitKey(1)& 0xFF
                        if k==27: #Escape KEY
                            im_out = imfill(image_binary, dim)
                            break              
                    cv2.destroyAllWindows()
                    im_out[im_out>0]=1
                    coord = np.where(im_out>0)
                    for nn in range(len(coord[0])):
                        mask[coord[0][nn],coord[1][nn]]=0
                    # plot data
                    fig = plt.figure(figsize=(14, 14))
                    ax1 = fig.add_subplot(121)
                    ax1.set_axis_off()
                    ax1.imshow(img_raw, cmap='gray')
                    ax2 = fig.add_subplot(122)
                    ax2.set_axis_off()
                    ax2.imshow(
                        color.label2rgb(mask.astype(np.uint8), img_raw, colors=[(255, 0, 0), (0, 255, 0), (255, 255, 0)], alpha=0.0008, bg_label=0,
                                        bg_color=None))
                    ax1.title.set_text('Raw_img')
                    ax2.title.set_text('Automated')
                    plt.show()
                    mask = mask.astype(np.uint8)
    
        cv2.destroyAllWindows()
pixel_size = data['pixel_size'][0]
data.close()

hdf5_file = h5py.File(os.path.join(input_path, 'post_proc.hdf5'), "w")
hdf5_file.create_dataset('mask', [len(MASK)] + [dim, dim], dtype=np.uint8)
hdf5_file.create_dataset('img_raw', [len(RAW)] + [dim, dim], dtype=np.float32)
dt = h5py.special_dtype(vlen=str)
hdf5_file.create_dataset('pixel_size', (1, 3), dtype=dt)
hdf5_file.create_dataset('phase', (1, 1), dtype=dt)

for i in range(len(MASK)):
    hdf5_file['mask'][i, ...] = MASK[i]
    hdf5_file['img_raw'][i, ...] = RAW[i]
hdf5_file['pixel_size'][...] = pixel_size
hdf5_file['phase'][...] = phase

hdf5_file.close()
