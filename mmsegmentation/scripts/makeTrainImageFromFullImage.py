import cv2
import os
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import ipdb

def max_conectArea(label):
    testa1 = measure.label(label, connectivity=2)  # 8连通
    # 如果想分别对每一个连通区域进行操作，比如计算面积、外接矩形、凸包面积等，则需要调用#measure子模块的regionprops（）函数
    props = measure.regionprops(testa1)

    numPix = []
    for ia in range(len(props)):
        numPix += [props[ia].area]

    # 像素最多的连通区域及其指引
    maxnum = max(numPix)
    index = numPix.index(maxnum)

    # 最大连通区域的bounding box
    start_x, start_y, end_x, end_y = props[index].bbox  # [minr, maxr),[minc, maxc)
    # 最大连通区域中的原始值
    w_half = min(end_x - start_x, end_y - start_y) / 2
    mid_x = (start_x + end_x) / 2
    mid_y = (start_y + end_y) / 2
    start_x = int(mid_x - w_half)
    start_y = int(mid_y - w_half)
    end_x = int(mid_x + w_half)
    end_y = int(mid_y + w_half)

    return start_x,end_x,start_y,end_y

if __name__ == "__main__":
    label_path = '/media/songcc/data/songcc/Retinal/dataset/REFUGE/OC_augGT_train+val'
    image_path = "/media/songcc/data/songcc/Retinal/dataset/REFUGE/OC_augImg_train+val"
    tar_label_path = "/media/songcc/data/songcc/Retinal/dataset/REFUGE/Aug_cup_pathes"
    tar_image_path = "/media/songcc/data/songcc/Retinal/dataset/REFUGE/Aug_cup_img_pathes"

    names = os.listdir(image_path)
    for name in names:
        img = cv2.imread(os.path.join(image_path,name))
        label_image = cv2.imread(os.path.join(label_path,name))
        label_image = cv2.cvtColor(label_image,cv2.COLOR_BGR2GRAY)
        
        w,h = np.shape(label_image)
        w_use = np.min([w,h])
        
        start_x,end_x,start_y,end_y = max_conectArea(label_image)
        center_x,center_y = (start_x + end_x) / 2, (start_y + end_y) / 2
        start_x = max(int(center_x - w_use // 5),0)
        end_x = min(int(center_x + w_use // 5),w)
        start_y = max(int(center_y - w_use // 5),0)
        end_y = min(int(center_y + w_use // 5),h)

        if (start_x == 0):
            end_x = w_use // 5 * 2
        elif (end_x == w):
            start_x = end_x - w_use // 5 * 2
        if (start_y == 0):
            end_y = w_use // 5 * 2
        elif (end_y == h):
            start_y = end_y - w_use // 5 * 2
        
        img_show = img[start_x:end_x,start_y:end_y,:]
        label = label_image[start_x:end_x,start_y:end_y]
        print(np.shape(label))
        # ipdb.set_trace()
        cv2.imwrite(os.path.join(tar_image_path,name),img_show)
        cv2.imwrite(os.path.join(tar_label_path,name),label * 255)
