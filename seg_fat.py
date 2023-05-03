"""
Created on Wed May  3 09:43:21 2023

@author: Marco Penso
"""
# segment a dicom image file
import scipy
import numpy as np
import cv2
import pydicom

X = []
Y = []

drawing=False # true if mouse is pressed
mode=True

def paint_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),1)
                cv2.line(image_binary,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),1)
                current_former_x = former_x
                current_former_y = former_y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),1)
            cv2.line(image_binary,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),1)
            current_former_x = former_x
            current_former_y = former_y
    return former_x,former_y


# path
input_path = r'G:/GRASSO/external_validation/5/seg/series1705-unknown/img0031--31.0386.dcm'
# read data
data_row_img = pydicom.dcmread(input_path)
img = data_row_img.pixel_array
print(img.shape, img.dtype)

tit=['---Segmenting FAT---']
img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
image_binary = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
cv2.namedWindow(tit[0])
cv2.setMouseCallback(tit[0],paint_draw)
while(1):
    cv2.imshow(tit[0],img)
    k=cv2.waitKey(1)& 0xFF
    if k==27: #Escape KEY
        im_out = image_binary
        break              
cv2.destroyAllWindows()
im_out[im_out>0]=255
coord = np.where((im_out)>1)
for i in range(len(coord[0])):
    img[coord[0][i],coord[1][i],1]=0

data_row_img.PixelData = img.tostring()
data_row_img.save_as(input_path)
