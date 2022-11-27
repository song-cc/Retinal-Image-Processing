import os 
import cv2
import numpy as np 

img_path = "/media/songcc/data/songcc/Retinal/DRIVE/test/png_images"
mask_path = "/media/songcc/data/songcc/Retinal/DRIVE/test/courseLabel"
tar_path = "/media/songcc/data/songcc/Retinal/DRIVE/test/ImageCourse"

names = os.listdir(img_path)
kernel = np.ones([15,15],np.uint8)
for name in names:
    img = cv2.imread(os.path.join(img_path,name))
    
    mask_img = cv2.imread(os.path.join(mask_path,name))
    mask_img = cv2.cvtColor(mask_img,cv2.COLOR_BGR2GRAY)
    mask_img = mask_img / 255
    mask_img = cv2.dilate(mask_img,kernel)

    img[:,:,0] = img[:,:,0] * mask_img
    img[:,:,1] = img[:,:,1] * mask_img
    img[:,:,2] = img[:,:,2] * mask_img

    print(np.max(img),name)
    cv2.imwrite(os.path.join(tar_path,name),img)