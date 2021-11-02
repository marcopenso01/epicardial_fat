# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:02:58 2021

@author: Marco Penso
"""

import os
import numpy as np
import h5py
import cv2
import pydicom # for reading dicom files
import matplotlib.pyplot as plt

X = []
Y = []

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


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(x,",",y)
        X.append(y)
        Y.append(x)
        cv2.destroyAllWindows()


def imfill(img):
    im_floodfill = img.copy()
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    return img | cv2.bitwise_not(im_floodfill)


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
    
    
def crop_or_pad_slice_to_size_specific_point(slice, nx, ny, cx, cy):
    
    if len(slice.shape) == 3:
        stack = [slice[:,:,0], slice[:,:,1], slice[:,:,2]]
        RGB = []
    else:
        stack = [slice]
        
    for i in range(len(stack)):
        img = stack[i]
        x, y = img.shape
        y1 = (cy - (ny // 2))
        y2 = (cy + (ny // 2))
        x1 = (cx - (nx // 2))
        x2 = (cx + (nx // 2))
    
        if y1 < 0:
            img = np.append(np.zeros((x, abs(y1))), img, axis=1)
            x, y = img.shape
            y1 = 0
        if x1 < 0:
            img = np.append(np.zeros((abs(x1), y)), img, axis=0)
            x, y = img.shape
            x1 = 0
        if y2 > 512:
            img = np.append(img, np.zeros((x, y2 - 512)), axis=1)
            x, y = img.shape
        if x2 > 512:
            img = np.append(img, np.zeros((x2 - 512, y)), axis=0)
    
        slice_cropped = img[x1:x1 + nx, y1:y1 + ny]
        if len(stack)>1:
            RGB.append(slice_cropped)
        
    if len(stack)>1:
        return np.dstack((RGB[0], RGB[1], RGB[2]))
    else:
        return slice_cropped
 


if __name__ == '__main__':

    # Paths settings
    input_folder = r'F:/FAT'
    nx = 160
    ny = 160
    force_overwrite = True
    fold = 'test'
    val_num = 0
    
    
    output_folder = os.path.join(input_folder, 'pre_processing')
    if not os.path.exists(output_folder) or force_overwrite:
        makefolder(output_folder)
    
    if len(os.listdir(output_folder)) == 0 or force_overwrite:

        print('This configuration of mode has not yet been preprocessed')
        print('Preprocessing now!')
        
        path = os.path.join(input_folder, fold)
    
        # ciclo su pazienti train
        MASK = []
        IMG_SEG = []  #img in uint8 con segmentazione
        IMG_RAW = []  #img in float senza segmentazione
        ROWS = []
        COLS = []
        PIXEL_SIZE = []
        IMG_UP = []
        IMG_DOWN = []
        IMG_LEFT = []
        IMG_RIGHT = []
        PAZ = []
        
        valMASK = []
        valIMG_SEG = []  #img in uint8 con segmentazione
        valIMG_RAW = []  #img in float senza segmentazione
        valROWS = []
        valCOLS = []
        valPIXEL_SIZE = []
        valIMG_UP = []
        valIMG_DOWN = []
        valIMG_LEFT = []
        valIMG_RIGHT = []
        valPAZ = []
        
        count = 0
        for paz, i in zip(os.listdir(path), range(len(os.listdir(path)))):
            
            if val_num !=0:
                if (i+1) % int(len(os.listdir(path)) / val_num) == 0 and len(count)<val_num:
                    val = True
                    count += 1
                    print('VALIDATION: processing paz %s' % paz)
                else:
                    val = False
                    print('TRAIN: processing paz %s' % paz)
            else:
                val = False
                print('TRAIN: processing paz %s' % paz)
            
            paz_path = os.path.join(path, paz)
            
            path_seg = os.path.join(paz_path, 'seg')
            if not os.path.exists(path_seg):
                raise Exception('path %s not found' % path_seg)
            path_seg = os.path.join(path_seg, os.listdir(path_seg)[0])
        
            path_raw = os.path.join(paz_path, 'raw')
            if not os.path.exists(path_raw):
                raise Exception('path %s not found' % path_raw)
            path_raw = os.path.join(path_raw, os.listdir(path_raw)[0])
            
            if len(os.listdir(path_seg)) != len(os.listdir(path_raw)):
                raise Exception('number of file in seg %s and row %s is not equal, for patient %s' % (len(os.listdir(path_seg)), len(os.listdir(path_raw)), paz))
            
            # dicom info                                                                                       
            dcmPath = os.path.join(path_raw, os.listdir(path_raw)[0])
            data_row_img = pydicom.dcmread(dcmPath)
            rows = int(data_row_img.Rows)
            cols = int(data_row_img.Columns)
            pixel_size  = [float(data_row_img.PixelSpacing[0]*(240/nx)),
                           float(data_row_img.PixelSpacing[1]*(240/ny)),
                           int(data_row_img.SpacingBetweenSlices)]
            
            # select center image
            X = []
            Y = []
            data_row_img = pydicom.dcmread(os.path.join(path_seg, os.listdir(path_seg)[120]))
            img = data_row_img.pixel_array
            cv2.imshow("image", img.astype('uint8'))
            cv2.namedWindow('image')
            cv2.setMouseCallback("image", click_event)
            cv2.waitKey(0)
            print(X)
            print(Y)
            
            # mask extraction
            for i in range(len(os.listdir(path_seg))):
                dcmPath = os.path.join(path_seg, os.listdir(path_seg)[i])
                data_row_img = pydicom.dcmread(dcmPath)
                img = data_row_img.pixel_array
                img = crop_or_pad_slice_to_size_specific_point(img, 240, 240, X[0], Y[0])
                temp_img = img.copy()
                    
                if len(np.argwhere(cv2.inRange(temp_img, (250, 250, 0), (255, 255, 0)))) > 10:
                    temp_img = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
                    for r in range(0, img.shape[0]):
                        for c in range(0, img.shape[1]):
                            if not img[r,c,0] == img[r,c,1] == img[r,c,2]:
                               temp_img[r,c]=255
    
                    mask = imfill(temp_img)
                    mask[mask>0]=1
                
                    if not mask.max()==1 or not mask.min()==0:
                        raise Exception('mask img %s has value max: %s, min: %s' % (os.listdir(path_seg)[i], mask.max(), mask.min()))
                    
                    
                    img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
                    mask = cv2.resize(mask, (nx, ny), interpolation=cv2.INTER_NEAREST)
                    
                    fig = plt.figure()
                    ax1 = fig.add_subplot(121)
                    ax1.imshow(img)
                    ax2 = fig.add_subplot(122)
                    ax2.imshow(mask)
                    plt.title('img %d' % (i+1));
                    plt.show()
                    
                    if val:
                        valMASK.append(mask)
                        valIMG_SEG.append(img)
                        valROWS.append(rows)
                        valCOLS.append(cols)
                        valPIXEL_SIZE.append(pixel_size)
                        valPAZ.append(paz)
                    else:
                        MASK.append(mask)
                        IMG_SEG.append(img)
                        ROWS.append(rows)
                        COLS.append(cols)
                        PIXEL_SIZE.append(pixel_size)
                        PAZ.append(paz)
                    
                    # save data raw
                    dcmPath = os.path.join(path_raw, os.listdir(path_raw)[i])
                    data_row_img = pydicom.dcmread(dcmPath)
                    img = data_row_img.pixel_array
                    img = crop_or_pad_slice_to_size_specific_point(img, 240, 240, X[0], Y[0])
                    img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
                    
                    if val:
                        valIMG_RAW.append(img)
                    else:
                        IMG_RAW.append(img)
                    
                    #save spatial image
                    dcmPath = os.path.join(path_raw, os.listdir(path_raw)[i-30])
                    data_row_img = pydicom.dcmread(dcmPath)
                    img = data_row_img.pixel_array
                    img = crop_or_pad_slice_to_size_specific_point(img, 240, 240, X[0], Y[0])
                    img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
                    
                    if val:
                        valIMG_UP.append(img)
                    else:
                        IMG_UP.append(img)
                    
                    dcmPath = os.path.join(path_raw, os.listdir(path_raw)[i+30])
                    data_row_img = pydicom.dcmread(dcmPath)
                    img = data_row_img.pixel_array
                    img = crop_or_pad_slice_to_size_specific_point(img, 240, 240, X[0], Y[0])
                    img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
                    
                    if val:
                        valIMG_DOWN.append(img)
                    else: 
                        IMG_DOWN.append(img)
                    
                    #save temporal image
                    pos_row = (i // 30)+1
                    pos_row = (i // 30)+1
                    ph1 = 30*(pos_row-1)
                    ph30 = 30*(pos_row)-1
                    vet = range(ph1, ph30+1)
                    
                    pos_col = i % 30
                    for _ in range(2):
                        pos_col -= 1
                        if pos_col < 0:
                            pos_col = 29  
                            
                    dcmPath = os.path.join(path_raw, os.listdir(path_raw)[vet[pos_col]])
                    data_row_img = pydicom.dcmread(dcmPath)
                    img = data_row_img.pixel_array
                    img = crop_or_pad_slice_to_size_specific_point(img, 240, 240, X[0], Y[0])
                    img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
                    
                    if val:
                        valIMG_LEFT.append(img)
                    else:
                        IMG_LEFT.append(img)
                            
                    pos_col = i % 30
                    for _ in range(2):
                        pos_col += 1
                        if pos_col > 29:
                            pos_col = 0
                    
                    dcmPath = os.path.join(path_raw, os.listdir(path_raw)[vet[pos_col]])
                    data_row_img = pydicom.dcmread(dcmPath)
                    img = data_row_img.pixel_array
                    img = crop_or_pad_slice_to_size_specific_point(img, 240, 240, X[0], Y[0])
                    img = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
                    
                    if val:
                        valIMG_RIGHT.append(img)
                    else:
                        IMG_RIGHT.append(img)
                    
                    '''
                    target_x=[]
                    target_y=[]
                    for i in range(len(PIXEL_SIZE)):
                        target_x.append(PIXEL_SIZE[i][0])
                        target_y.append(PIXEL_SIZE[i][1])
                    np.asarray(target_x).min()
                    '''
        
        hdf5_file = h5py.File(os.path.join(output_folder, 'train.hdf5'), "w")
                 
        dt = h5py.special_dtype(vlen=str)
        hdf5_file.create_dataset('paz', (len(PAZ),), dtype=dt)
        hdf5_file.create_dataset('pixel_size', (len(PIXEL_SIZE),3), dtype=dt)
        hdf5_file.create_dataset('mask', [len(MASK)] + [nx, ny], dtype=np.uint8)
        hdf5_file.create_dataset('img_seg', [len(IMG_SEG)] + [nx, ny, 3], dtype=np.uint8)
        hdf5_file.create_dataset('img_raw', [len(IMG_RAW)] + [nx, ny], dtype=np.float32)
        hdf5_file.create_dataset('img_up', [len(IMG_UP)] + [nx, ny], dtype=np.float32)
        hdf5_file.create_dataset('img_down', [len(IMG_DOWN)] + [nx, ny], dtype=np.float32)
        hdf5_file.create_dataset('img_left', [len(IMG_LEFT)] + [nx, ny], dtype=np.float32)
        hdf5_file.create_dataset('img_right', [len(IMG_RIGHT)] + [nx, ny], dtype=np.float32)
        
        for i in range(len(PAZ)):
             hdf5_file['paz'][i, ...] = PAZ[i]
             hdf5_file['pixel_size'][i, ...] = PIXEL_SIZE[i]
             hdf5_file['mask'][i, ...] = MASK[i]
             hdf5_file['img_seg'][i, ...] = IMG_SEG[i]
             hdf5_file['img_raw'][i, ...] = IMG_RAW[i]
             hdf5_file['img_up'][i, ...] = IMG_UP[i]
             hdf5_file['img_down'][i, ...] = IMG_DOWN[i]
             hdf5_file['img_left'][i, ...] = IMG_LEFT[i]
             hdf5_file['img_right'][i, ...] = IMG_RIGHT[i]
        
        # After loop:
        hdf5_file.close()
        
        if val_num !=0:
            
            hdf5_file = h5py.File(os.path.join(output_folder, 'val.hdf5'), "w")
                 
            dt = h5py.special_dtype(vlen=str)
            hdf5_file.create_dataset('paz', (len(valPAZ),), dtype=dt)
            hdf5_file.create_dataset('pixel_size', (len(valPIXEL_SIZE),3), dtype=dt)
            hdf5_file.create_dataset('mask', [len(valMASK)] + [nx, ny], dtype=np.uint8)
            hdf5_file.create_dataset('img_seg', [len(valIMG_SEG)] + [nx, ny, 3], dtype=np.uint8)
            hdf5_file.create_dataset('img_raw', [len(valIMG_RAW)] + [nx, ny], dtype=np.float32)
            hdf5_file.create_dataset('img_up', [len(valIMG_UP)] + [nx, ny], dtype=np.float32)
            hdf5_file.create_dataset('img_down', [len(valIMG_DOWN)] + [nx, ny], dtype=np.float32)
            hdf5_file.create_dataset('img_left', [len(valIMG_LEFT)] + [nx, ny], dtype=np.float32)
            hdf5_file.create_dataset('img_right', [len(valIMG_RIGHT)] + [nx, ny], dtype=np.float32)
            
            for i in range(len(PAZ)):
                 hdf5_file['paz'][i, ...] = valPAZ[i]
                 hdf5_file['pixel_size'][i, ...] = valPIXEL_SIZE[i]
                 hdf5_file['mask'][i, ...] = valMASK[i]
                 hdf5_file['img_seg'][i, ...] = valIMG_SEG[i]
                 hdf5_file['img_raw'][i, ...] = valIMG_RAW[i]
                 hdf5_file['img_up'][i, ...] = valIMG_UP[i]
                 hdf5_file['img_down'][i, ...] = valIMG_DOWN[i]
                 hdf5_file['img_left'][i, ...] = valIMG_LEFT[i]
                 hdf5_file['img_right'][i, ...] = valIMG_RIGHT[i]
            
            # After loop:
            hdf5_file.close()

    else:

        print('Already preprocessed this configuration. Loading now!')



'''
pixels = np.argwhere(mask)
a = []
for i in range(len(pixels)):
    a.append(img[pixels[i][0], pixels[i][1]])
a = sorted(np.asarray(a))
'''
