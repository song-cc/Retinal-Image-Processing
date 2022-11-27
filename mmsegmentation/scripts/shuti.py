import os 
import cv2
import numpy as np 
from skimage import measure
import ipdb

path = "/media/songcc/data/songcc/Retinal/AVSegmentation_data/DRIVE_AV/test/1st_manual"
tar_path = "/media/songcc/data/songcc/Retinal/AVSegmentation_data/DRIVE_AV/test/gt_eval"

if not os.path.isdir(tar_path):
    os.makedirs(tar_path)

names = os.listdir(path)

for name in names:
    img = cv2.imread(os.path.join(path,name))
    img[img[:,:,1] == 255] = [255,0,0]
    cv2.imwrite(os.path.join(tar_path,name[:2] + ".png"),img)

# AV_path = "/media/songcc/data/songcc/Retinal/AVSegmentation_data/DRIVE_AV/test/SA_Test_post"
# img_path = "/media/songcc/data/songcc/Retinal/AVSegmentation_data/DRIVE_AV/test/images"
# tar_label_path = "/media/songcc/data/songcc/Retinal/AVSegmentation_data/DRIVE_AV/test/label_train_used"
# tar_AV_path = "/media/songcc/data/songcc/Retinal/AVSegmentation_data/DRIVE_AV/test/AV_train_used"
# tar_image_path = "/media/songcc/data/songcc/Retinal/AVSegmentation_data/DRIVE_AV/test/processImage_train_used"

# def connect_analysis(pred):
#     _,pred = cv2.threshold(pred,10,255,cv2.THRESH_BINARY)

#     label = measure.label(pred)
#     region_props = measure.regionprops(label)
#     n = len(region_props)
#     max_region = region_props[0].label
#     index = 0

#     for i in range(n):
#         # ipdb.set_trace()
#         if region_props[i].area > max_region:
#             max_region = region_props[i].area
#             index = i
    
#     return region_props[index].bbox

# if not os.path.isdir(tar_label_path):
#     os.makedirs(tar_label_path)
# if not os.path.isdir(tar_AV_path):
#     os.makedirs(tar_AV_path)
# if not os.path.isdir(tar_image_path):
#     os.makedirs(tar_image_path)

# names = os.listdir(img_path)
# for name in names:
#     img = cv2.imread(os.path.join(img_path,name))
#     img0 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     name = name[:2] + ".png"
#     tar_name = name
#     bbox = connect_analysis(img0)
#     top,left,down,right = bbox[0],bbox[1],bbox[2],bbox[3]
#     img = img[top:down,left:right,:]
#     av_img = cv2.imread(os.path.join(AV_path,name))
#     av_img = cv2.cvtColor(av_img,cv2.COLOR_BGR2GRAY)
#     av_img = av_img[top:down,left:right]
#     # label_img = av_img.copy()
#     # label_img[av_img > 0] = 1
#     # label_img = label_img.astype(np.uint8)
 
#     cv2.imwrite(os.path.join(tar_av_path,tar_name),av_img)
#     # cv2.imwrite(os.path.join(tar_AV_path,tar_name),av_img)
#     # cv2.imwrite(os.path.join(tar_label_path,tar_name),label_img)
