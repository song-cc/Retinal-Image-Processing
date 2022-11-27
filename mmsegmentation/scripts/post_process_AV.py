import os
import cv2
import numpy as np
from skimage import measure
import ipdb

def connect_analysis(pred):
    _,pred = cv2.threshold(pred,0.5,255,cv2.THRESH_BINARY)
    h,w = np.shape(pred)

    label = measure.label(pred)
    region_props = measure.regionprops(label)
    result = np.zeros((h,w),dtype=np.uint8)
    res = np.ones((h,w),dtype=np.uint8)

    for region_prop in region_props:
        if region_prop.area > 30:
            result[label == region_prop.label] = res[label == region_prop.label]*255
    
    return result

def got_AV(vessel,vein):
    vein = connect_analysis(vein)
    kernel = np.ones((5,5),np.uint8)
    vein = vein / 255
    vessel = vessel / 255
    vein = vein * vessel * 255
    vessel = vessel * 255
    vein = cv2.dilate(vein,kernel,iterations = 1)
    vein[vein > 10] = 255
    vein[vein < 10] = 0

    vessel,vein = vessel.astype(np.int16),vein.astype(np.int16)
    artery = vessel - vein
    h,w = np.shape(artery)
    res = np.zeros([h,w,3])
    
    artery[artery > 1] = 255
    artery[artery <= 1] = 0
    
    vein = vessel - artery
    _,vein = cv2.threshold(vein,100,255,cv2.THRESH_BINARY)
    res[artery > 1] = [0,0,255]
    res[vein > 1] = [255,0,0]

    res = res.astype(np.uint8)

    return res

if __name__ == "__main__":
    vessel_path = "/media/songcc/data/songcc/Retinal/AVSegmentation_data/DRIVE_AV/test/SA_Test_post"
    vein_path = "work_dirs/drive0"
    tar_path = "work_dirs/drive_AV"
    if not os.path.isdir(tar_path):
        os.makedirs(tar_path)

    names = os.listdir(vessel_path)
    for name in names:
        print(name)
        vessel_img = cv2.imread(os.path.join(vessel_path,name))
        vessel_img = cv2.cvtColor(vessel_img,cv2.COLOR_BGR2GRAY)

        vein_img = cv2.imread(os.path.join(vein_path,name))
        h,w,_ = np.shape(vein_img)
        res = np.zeros([h,w]).astype(np.uint8)
        res[vein_img[:,:,0] == 255] = 255
        vein_img = res 
        # vein_img = cv2.cvtColor(vein_img,cv2.COLOR_BGR2GRAY)

        res = got_AV(vessel_img,vein_img)
        cv2.imwrite(os.path.join(tar_path,name),res)