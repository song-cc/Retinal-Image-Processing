import cv2
import os
import numpy as np

img_path = "/media/songcc/data/songcc/Retinal/AVSegmentation_data/DRIVE_AV/training/AV_train_used"
SA_path = "/media/songcc/data/songcc/Retinal/AVSegmentation_data/DRIVE_AV/training/SA_UNet-train_post"
tar_path = "/media/songcc/data/songcc/Retinal/AVSegmentation_data/DRIVE_AV/training/processedAV_train_used"

if not os.path.isdir(tar_path):
    os.makedirs(tar_path)

names = os.listdir(img_path)

for name in names:
    img = cv2.imread(os.path.join(img_path,name))
    mask = cv2.imread(os.path.join(SA_path,name))
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    mask[mask < 100] = 0
    mask[mask > 100] = 1

    img[:,:,0] = np.multiply(img[:,:,0],mask)
    img[:,:,1] = np.multiply(img[:,:,1],mask)
    img[:,:,2] = np.multiply(img[:,:,2],mask)
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(tar_path,name),img)
    