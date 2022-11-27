import os
import cv2
import numpy as np
from skimage import measure
from random import randint

def connect_analysis(pred):
    _,pred = cv2.threshold(pred,0.5,255,cv2.THRESH_BINARY)
    h,w = np.shape(pred)

    label = measure.label(pred)
    region_props = measure.regionprops(label)
    result = np.zeros((h,w),dtype=np.uint8)
    res = np.ones((h,w),dtype=np.uint8)

    for region_prop in region_props:
        if region_prop.area > 50:
            result[label == region_prop.label] = res[label == region_prop.label]*255
    
    return result

if __name__ == "__main__":
    path = "work_dirs/drive0"
    tar_path = "work_dirs/drive0"
    if not os.path.isdir(tar_path):
        os.makedirs(tar_path)

    names = os.listdir(path)[:20]
    for name in names:
        img = cv2.imread(os.path.join(path,name))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        res = connect_analysis(img)
        cv2.imwrite(os.path.join(tar_path,name),res)