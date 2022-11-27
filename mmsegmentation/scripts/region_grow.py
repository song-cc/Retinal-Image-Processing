import cv2
import os
import numpy as np

max_rate = 1.05
min_rate = 0.95
def region_grow(im,mask):
    im_shape = im.shape
    height = im_shape[0]
    width = im_shape[1]

    class Point(object):
        def __init__(self , x , y):
            self.x = x
            self.y = y
        def getX(self):
            return self.x
        def getY(self):
            return self.y
    connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
    #标记，判断种子是否已经生长
    img_mark = np.zeros_like(mask)

    seed_list = []
    bgr = [0] * 3
    n = 0
    for i in range(height):
        for j in range(width):
            if mask[i,j] >= 1:
                seed_list.append(Point(i,j))
                bgr[0] += im[i,j,0]
                bgr[1] += im[i,j,1]
                bgr[2] += im[i,j,2]
                n += 1
    
    bgr[0] = bgr[0] / n
    bgr[1] = bgr[1] / n
    bgr[2] = bgr[2] / n

    class_k = 1#类别
    while (len(seed_list) > 0):
        seed_tmp = seed_list[0]
        #将以生长的点从一个类的种子点列表中删除
        seed_list.pop(0)

        img_mark[seed_tmp.x, seed_tmp.y] = class_k

        # 遍历8邻域
        for i in range(4):
            tmpX = seed_tmp.x + connects[i].x
            tmpY = seed_tmp.y + connects[i].y

            if (tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= width):
                continue
            #在种子集合中满足条件的点进行生长
            if (img_mark[tmpX, tmpY] == 0 and bgr[0] * max_rate > im[tmpX,tmpY,0] > bgr[0] * min_rate and bgr[1] * max_rate > im[tmpX,tmpY,1] > bgr[1] * min_rate
            and bgr[2] * max_rate > im[tmpX,tmpY,2] > bgr[2] * min_rate):
                img_mark[tmpX, tmpY] = class_k
                seed_list.append(Point(tmpX, tmpY))
    
    return img_mark * 255

if __name__ == "__main__":
    mask_path = "/media/songcc/data/songcc/Retinal/code/SA-UNet/Test_2modul"
    img_path = "/media/songcc/data/songcc/Retinal/DRIVE/test/png_images"
    tar_path = "/media/songcc/data/songcc/Retinal/code/SA-UNet/Test_region"
    if not os.path.isdir(tar_path):
        os.makedirs(tar_path)

    names = os.listdir(mask_path)

    for name in names:
        im = cv2.imread(os.path.join(img_path,name))
        mask = cv2.imread(os.path.join(mask_path,name))
        mask = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
        vessel = region_grow(im,mask)
        cv2.imwrite(os.path.join(tar_path,name),vessel)