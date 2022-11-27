import cv2
import numpy as np
from skimage import morphology
from skimage import measure
from scipy.signal import convolve2d
from image_process import vesselCal_preprocess
from skimage.measure import regionprops
from inflections import *
import math
import ipdb
def got_OD_diameter(OD):
    '''
       OD: np.array;   The OD segmentation image
    '''
    edge = cv2.Canny(OD,10,150)
    
    y,x = np.nonzero(edge)
    edge_list = np.array([[_x,_y] for _x,_y in zip(x,y)])

    ellipse = cv2.fitEllipse(edge_list)
    return max(ellipse[1]),ellipse[0]

def got_vessel_diameter(vessel):
    _,vessel = cv2.threshold(vessel,10,255,cv2.THRESH_BINARY)
    vessel = vessel / 255
    thinVessel = morphology.skeletonize(vessel)
    node = got_vessel_node(thinVessel)
    vessel = vessel.astype(np.int)
    node = node.astype(np.int)

    vessel_seg = vessel - node
    thinVessel_seg = thinVessel - node
    vessel_seg[vessel_seg < 0] = 0
    thinVessel_seg[thinVessel_seg < 0] = 0

def got_vessel_seg(vessel):
    if len(np.shape(vessel)) == 3:
        vessel = cv2.cvtColor(vessel,cv2.COLOR_RGB2GRAY)
    if np.max(vessel) < 20:
        vessel = vessel * 255
    _,vessel = cv2.threshold(vessel,10,255,cv2.THRESH_BINARY)
    vessel = vessel / 255
    thinVessel = morphology.skeletonize(vessel)
    node = got_vessel_node(thinVessel)
    vessel = vessel.astype(np.int)
    node = node.astype(np.int)

    vessel_seg = vessel - node

    return vessel_seg

def got_vessel_node(thinVessel):
    kernel_1 = np.ones((3,3))
    kernel_2 = np.zeros((3,3))
    kernel_2[1,1] = 1
    node = convolve2d(thinVessel,kernel_1,mode="same")
    node_center = convolve2d(thinVessel,kernel_2,mode="same")
    node = np.multiply(node,node_center)
    node_res = np.zeros_like(node).astype(np.uint8)
    node_res[node > 3] = 1

    kernel = np.ones((12,12),np.uint8)
    node_res = cv2.dilate(node_res,kernel,iterations = 1)
    
    return node_res

def got_one_width(vessel,boneLine):
    edge = cv2.Canny(vessel,10,150)
    y,x = np.nonzero(edge)
    y_target,x_target = np.nonzero(boneLine)
    
    len_y0 = np.shape(y)[0]
    len_yTarget = np.shape(y_target)[0]
    dises = []

    for k in range(len_y0):
        dis = 10000
        for i in range(len_yTarget):
            y0  = y_target[i]
            x0 = x_target[i]
            dis = min(dis,((y[k] - y0) ** 2 + (x[k] - x0) ** 2) ** 0.5)
        dises.append(dis)

    return np.mean(dises) * 2

def got_every_vessel(seg_vessel):
    if len(np.shape(seg_vessel)) == 3:
        seg_vessel = cv2.cvtColor(seg_vessel,cv2.COLOR_RGB2GRAY)
    if np.max(seg_vessel) < 20:
        seg_vessel = seg_vessel * 255
    _,pred = cv2.threshold(seg_vessel,0.5,255,cv2.THRESH_BINARY)
    h,w = np.shape(pred)

    label = measure.label(pred)
    region_props = measure.regionprops(label)
    vessels = []

    for region_prop in region_props:
        res = np.ones((h,w),dtype=np.uint8)
        res[label == region_prop.label] = 255
        bbox = region_prop.bbox

        vessel_ele = res[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        
        vessels.append(vessel_ele)
    
    return vessels

def got_CRAEandCRVE(vein_diameters,artery_diameters,OD_diameters):
    
    CRVE = 0.95 * (((np.max(vein_diameters)) ** 2 + (np.min(vein_diameters)) ** 2) ** 0.5)
    CRVE = CRVE * 1800 / OD_diameters

    CRAE = 0.88 * (((np.max(artery_diameters)) ** 2 + (np.min(artery_diameters)) ** 2) ** 0.5)
    CRAE = CRAE * 1800 / OD_diameters

    return CRAE,CRVE

def region_grow(im,seed):
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
    img_mark = np.zeros_like(im)

    seed_list = []
    
    seed_list.append(Point(seed[0],seed[1]))
    
    while (len(seed_list) > 0):
        seed_tmp = seed_list[0]
        #将以生长的点从一个类的种子点列表中删除
        seed_list.pop(0)

        img_mark[seed_tmp.x, seed_tmp.y] = 1

        # 遍历8邻域
        for i in range(4):
            tmpX = seed_tmp.x + connects[i].x
            tmpY = seed_tmp.y + connects[i].y

            if (tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= width):
                continue
            #在种子集合中满足条件的点进行生长
            if (img_mark[tmpX, tmpY] == 0 and im[tmpX,tmpY] > 0):
                img_mark[tmpX, tmpY] = 1
                seed_list.append(Point(tmpX, tmpY))
    
    return img_mark * 255

def got_angle_reference(img):
    img_pre = vesselCal_preprocess()
    image = img_pre.border(img, 592)
    skel = img_pre.img_skeltonize(image)

    corners = img_pre.angle_coordination_detect(skel.copy())
    r, c = image.shape

    angle_list = []
    for i in range(len(corners)):
        center_coordination = corners[i].ravel()
        x = center_coordination[0]
        y = center_coordination[1]

        back = np.zeros((r, c), dtype=np.uint8)
        roi = skel[y - 10:y + 10, x - 10:x + 10]
        back[y - 10:y + 10, x - 10:x + 10] = roi

        corners_2 = branching_angle_ext(back)
        angle_value = branching_coordination_ext(corners_2, center_coordination)

        if angle_value != None:
            angle_list.append(angle_value)

    return np.mean(angle_list)

def branching_angle_ext(back):
    '''
    计算中心角点区域内的角点
    '''
    img_pre = vesselCal_preprocess()
    corners_2 = img_pre.angle_coordination_detect(back)

    return corners_2

def branching_coordination_ext(corners_2,center_coordination):
    '''
    计算中心角点区域内，所有角点与中心角点的夹角，取与中心角点夹角的最小值作为该分枝角
    :return:  该分枝角的角度
    '''
    angle = []
    angle_point = []
    point_1 = center_coordination

    if len(corners_2) == 4:
        for corner in corners_2:
            a = corner.ravel()
            if  np.array_equal(a,center_coordination) == True:
                point_1 =  center_coordination
            else:
                angle_point.append(a)
        angle.append(branching_angle_calc(point_1,angle_point[0],angle_point[1]))
        angle.append(branching_angle_calc(point_1,angle_point[0],angle_point[2]))
        angle.append(branching_angle_calc(point_1,angle_point[1],angle_point[2]))

        angle_value = min(angle)
        angle.clear()
        return angle_value

    elif len(corners_2) == 3:
        for corner in corners_2:
            a = corner.ravel()
            if  np.array_equal(a,center_coordination) == True:
                point_1 = center_coordination
            else:
                angle_point.append(corner.ravel())
        angle.append(branching_angle_calc(point_1,angle_point[0],angle_point[1]))

        angle_value = min(angle)
        if (angle_value > 90)&(angle_value <= 180):
            angle_value = 180 - angle_value
        else:
            angle_value = angle_value
        angle.clear()
        return angle_value

def branching_angle_calc(point_1,point_2,point_3):
    """
        根据三点坐标计算夹角
        :param point_1: 点1坐标
        :param point_2: 点2坐标
        :param point_3: 点3坐标
        :return: 返回任意角的夹角值，这里只是返回点1的夹角
    """
    angle_degree  = []
    angle_degree.clear()
    a = math.sqrt(
        (point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (point_2[1] - point_3[1]))
    b = math.sqrt(
        (point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (point_1[1] - point_3[1]))
    c = math.sqrt(
        (point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (point_1[1] - point_2[1]))
    
    if abs((a * a - b * b - c * c) / (-2 * b * c)) > 1:
        A = 0
    else:
        A = math.degrees(math.acos((a * a - b * b - c * c) / (-2 * b * c)))
    if abs((b * b - a * a - c * c) / (-2 * a * c)) > 1 :
        B = 0
    else:
        B = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
    if abs((c * c - a * a - b * b) / (-2 * a * b))>1:
        C = 0
    else:
        C = math.degrees(math.acos((c * c - a * a - b * b) / (-2 * a * b)))

    return  A

def got_curvature(img):
    img_pre = vesselCal_preprocess()
    image = img_pre.border(img, 592)
    skel = img_pre.img_skeltonize(image)
    labels = img_pre.corner_detect(skel)
    regions = regionprops(labels) 

    torts_list = []
    arc_torts_list = []
    arcbased_list = []
    for region in regions:
        blob_skeleton_stack = img_pre.curve_ext(region,labels)
        count = np.count_nonzero(blob_skeleton_stack)

        if count > 30 :
            torts, arcbased, arc_torts = tortuosity(blob_skeleton_stack)
            torts_list.append(torts)
            arc_torts_list.append(arc_torts)
            arcbased_list.append(arcbased)

    torts_value = (180/ np.mean(torts_list))
    art_torts_value = np.mean(arc_torts_list)
    arcbased_value = np.mean(arcbased_list)

    return  torts_value, art_torts_value, arcbased_value

def arclength(order):
    h = distance(order[0],order[-1])

    arclength = 0
    n = len(order)
    for i in range(n-1):
        arclength+= distance(order[i],order[i+1])
    
    t = arclength/h

    return t

def tortuosity(img):
    #cv2.imshow("Contours_plt", output)
    #cv2.waitKey(0)
    count = np.count_nonzero(img)

    if count >= 30:
        x, y, h, w = cv2.boundingRect(img)  # x,y 表示矩形框的坐标 ，h,w 表示轮廓的宽和高

        roi = img[y:y + w, x:x + h]

        r, c = roi.shape
        #cv2.imshow("roi", roi)
        back = np.zeros((r + 2, c + 2), dtype=np.uint8)
        back[1:r + 1, 1:c + 1] = roi

        iroi = back.copy()
        #cv2.imshow('iroi', iroi)
        #cv2.waitKey(0)
        order = order_points(iroi)  # 只统计了一个曲线的所有点坐标

        # ************************进行折线化，返回折线的像素点队列****************************
        inflections = contour_inflections(iroi)
        clean = []
        [clean.append(x) for x in inflections if x not in clean]

        angles = get_angles(inflections)
        c_angles = [incom for incom in angles if str(incom) != 'nan']

        torts = np.mean(c_angles)
        arcbased = arclength(order)
        arc_torts= arcbased
        return torts, arcbased, arc_torts

def got_fractal_dimension(image):
    """ Calculates the fractal dimension of an image represented by a 2D numpy array.

    The algorithm is a modified box-counting algorithm as described by Wen-Li Lee and Kai-Sheng Hsieh.

    Args:
        image: A 2D array containing a grayscale image. Format should be equivalent to cv2.imread(flags=0).
               The size of the image has no constraints, but it needs to be square (m×m array).
    Returns:
        D: The fractal dimension Df, as estimated by the modified box-counting algorithm.
    """
    M = image.shape[0]  # image shape
    G_min = image.min()  # lowest gray level (0=white)
    G_max = image.max()  # highest gray level (255=black)
    G = G_max - G_min + 1  # number of gray levels, typically 256
    prev = -1  # used to check for plateaus
    r_Nr = []

    for L in range(2, (M // 2) + 1):
        #循环变量由2-256
        M_h = G//(M //L)
        h = max(1, M_h )  # minimum box height is 1
        N_r = 0
        r = L / M
        for i in range(0, M, L):
            #循环变量范围：0-512，递增L--
            boxes = [[]] * ((G + h - 1) // h) 
            for row in image[i:i + L]:  # boxes that exceed bounds are shrunk to fit
                for pixel in row[i:i + L]:
                    height = (pixel - G_min) // h  # lowest box is at G_min and each is h gray levels tall
                    boxes[height].append(pixel)  # assign the pixel intensity to the correct box
            stddev = np.sqrt(np.var(boxes, axis=1))  # calculate the standard deviation of each box
            stddev = stddev[~np.isnan(stddev)]  # remove boxes with NaN standard deviations (empty)
            nBox_r = 2 * (stddev // h) + 1
            N_r += sum(nBox_r)
        if N_r != prev:  # check for plateauing
            r_Nr.append([r, N_r])
            prev = N_r
    x = np.array([np.log(1 / point[0]) for point in r_Nr])  # log(1/r)
    y = np.array([np.log(point[1]) for point in r_Nr])  # log(Nr)
    D = np.polyfit(x, y, 1)[0]  # D = lim r -> 0 log(Nr)/log(1/r)
    return D

if __name__ == "__main__":
    vessel = cv2.imread("vessel_seg.png")
    vessel = cv2.cvtColor(vessel,cv2.COLOR_BGR2GRAY)
    
    seed = [453,163]
    res = region_grow(vessel,seed)
    cv2.imwrite("res.png",res)
