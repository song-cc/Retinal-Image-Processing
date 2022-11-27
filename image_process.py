import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label,regionprops

import  cv2
import  numpy as np
from skimage import  morphology,img_as_bool

class vesselCal_preprocess(object):

    def __int__(self,):
        self.img = np.ndarray(())

    def border(self,image,size):
        x = cv2.resize(image, (size, size))
        return x

    def img_skeltonize(self,img):

        _, thresh = cv2.threshold(img, 0.5 ,255, cv2.THRESH_BINARY)

        imbool = img_as_bool(thresh)
        skel = morphology.skeletonize(imbool)
        skel.dtype = np.uint8
        skel = skel * 255

        return skel

    def angle_coordination_detect(self, skel):
        # *****************************角点检测***********************************
        corners = cv2.goodFeaturesToTrack(skel, 20, 0.001, 10)
        # corners = cv2.cornerHarris(skel, 2, 3, 0.04)
        corners = np.int0(corners)
        # for j in corners:
        #    x_2, y_2 = j.ravel()
        return corners

    def angle_detect_seg(self, skel):
        # *****************************角点检测***********************************
        corners = cv2.goodFeaturesToTrack(skel, 20, 0.001, 10)
        corners = np.int0(corners)

        for i in corners:
            x, y = i.ravel()
            image_1 = cv2.circle(skel, (x, y), 4, (0, 0, 255), -1)
        # *******************************8连通域检测*******************************
        ret, labels = cv2.connectedComponents(image_1)

        return labels

    def  corner_detect(self,skel):
        # *****************************角点检测***********************************
        corners = cv2.goodFeaturesToTrack(skel, 20, 0.001, 10)
        corners = np.int0(corners)

        for i in corners:
            x, y = i.ravel()
            image_1 = cv2.circle(skel, (x, y), 4, (0, 0, 255), -1)

        # *******************************8连通域检测*******************************
        _, labels = cv2.connectedComponents(image_1)

        return  labels


    def  curve_ext(self,region,labels):

        blob_skeleton = labels == region.label
        blob_skeleton_stack = np.uint8(np.dstack((blob_skeleton, blob_skeleton, blob_skeleton))) * 255
        skel_pixels = np.where(blob_skeleton)  # address_index point

        for k in range(len(skel_pixels[0])):
            if np.sum(blob_skeleton[skel_pixels[0][k] - 1:skel_pixels[0][k] + 2,skel_pixels[1][k] - 1:skel_pixels[1][k] + 2]) == 2:
                blob_skeleton_stack[skel_pixels[0][k], skel_pixels[1][k]] = [255, 0, 0]

            elif np.sum(blob_skeleton[skel_pixels[0][k] - 1:skel_pixels[0][k] + 2,skel_pixels[1][k] - 1:skel_pixels[1][k] + 2]) == 4:
                blob_skeleton_stack[skel_pixels[0][k], skel_pixels[1][k]] = [0, 255, 0]
            elif np.sum(blob_skeleton[skel_pixels[0][k] - 1:skel_pixels[0][k] + 2,skel_pixels[1][k] - 1:skel_pixels[1][k] + 2]) > 4:
                blob_skeleton_stack[skel_pixels[0][k], skel_pixels[1][k]] = [0, 0, 255]

        blob_skeleton_stack = cv2.cvtColor(blob_skeleton_stack, cv2.COLOR_BGR2GRAY)
        blob_skeleton_stack[blob_skeleton_stack < 255] = 0

        return blob_skeleton_stack

def got_max_connect(img_ori):
    """
    :param img: it could be rgb image
    :return:
    """
    if len(np.shape(img_ori)) == 3:
        img = cv2.cvtColor(img_ori,cv2.COLOR_RGB2GRAY)
    else:
        img = img_ori
    _,binary_img = cv2.threshold(img,10,255,cv2.THRESH_BINARY)

    label_img = label(binary_img)
    props = regionprops(label_img)

    nums = len(props)
    max_connect_area = 0
    index = 0
    for i in range(nums):
        if max_connect_area < props[i].area:
            index = i
            max_connect_area = props[i].area

    bbox = props[index].bbox
    
    if len(np.shape(img_ori)) == 3:
        return img_ori[bbox[0]:bbox[2],bbox[1]:bbox[3],:]
    else:
        return img_ori[bbox[0]:bbox[2],bbox[1]:bbox[3]]

def got_connect_bbox(mask):
    _, binary_img = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    label_img = label(binary_img)
    props = regionprops(label_img)

    res = []
    nums = len(props)
    index = 0
    for i in range(nums):
        res.append(props[i].bbox)

    return res

def load_process(img,w_in,h_in):
    """
    在加载图像时做的操作
    :param img:
    :return:
    """
    img = got_max_connect(img)
    img_shape = np.shape(img)

    w_use = np.max(img_shape)
    img_res = np.zeros([w_use,w_use,3])
    img_res = img_res.astype(np.uint8)

    center_res = w_use // 2
    start_x = center_res - img_shape[0] // 2
    end_x = img_shape[0] + start_x
    start_y = center_res - img_shape[1] // 2
    end_y = img_shape[1] + start_y

    img_res[start_x:end_x,start_y:end_y,:] = img
    img_res  = cv2.resize(img_res,dsize=(w_in,h_in))
    return img_res

def reszie(ori_img,center_loc,size):
    """
    在原图上进行裁剪，以center_loc为中心，裁剪size大小的图像
    :param ori_img:
    :param center_loc: 目标图像中心相在ori_img的坐标
    :param size: 目标图像的尺寸
    :return:
    """

    ori_shape = np.shape(ori_img)
    start_x = center_loc[0] - size[0] // 2
    end_x = start_x + size[0]
    start_y = center_loc[1] - size[1] // 2
    end_y = start_y + size[1]

    if start_x <= 0:
        start_x = 0
        end_x = size[0]
    if start_y <= 0:
        start_y = 0
        end_y = size[1]
    if end_y >= ori_shape[1]:
        start_y = ori_shape[1] - size[1]
        end_y = ori_shape[1]
    if end_x >= ori_shape[0]:
        start_x = ori_shape[0] - size[0]
        end_x = ori_shape[0]

    return ori_img[start_x:end_x,start_y:end_y,:],[(start_x + end_x) // 2,(start_y + end_y) // 2]

def got_Roi_image(img,mask,w_ratio = 0.3):
    bboxes = got_connect_bbox(mask)
    nums = len(bboxes)

    w,h = np.shape(mask)
    w_tar,h_tar = int(w * w_ratio),int(h * w_ratio)

    res = []
    roi_index = []
    for i in range(nums):
        bbox = bboxes[i]
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2

        w_half,h_half = w_tar // 2 , h_tar // 2
        start_x = int(center_x - w_half)
        end_x = int(start_x + w_tar)

        start_y = int(center_y - h_half)
        end_y = int(start_y + h_tar)
        if start_x < 0:
            start_x == 0
            end_x == w_tar
        if start_y < 0:
            start_x == 0
            end_x == h_tar
        if end_x >= w:
            start_x = w - w_tar
            end_x = w - 1
        if end_y >= h:
            start_y = h - h_tar
            end_y = end_y - 1
        imshow = img[start_x:end_x,start_y:end_y,:]
        roi_index.append([start_x,end_x,start_y,end_y])
        res.append(imshow)

    return res,roi_index