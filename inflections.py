import cv2
import numpy as np
import math

def distance(a,b):
    res = (b[0]-a[0])**2 + (b[1]-a[1])**2
    res = math.sqrt(res)

    return res

def dfs(mat,i,j):

    visit[i][j] = 1

    if (j!=0 and visit[i][j-1]==0 and mat[i][j-1]!=0):
        return (i,j-1)

    if (j+1<c and visit[i][j+1]==0 and mat[i][j+1]!=0):
        return (i,j+1)

    if (i-1>=0 and visit[i-1][j]==0 and mat[i-1][j]!=0):
        return (i-1,j)
    
    if (i+1<r and  visit[i+1][j]==0 and mat[i+1][j]!=0):
        return (i+1,j)

    if (i-1>=0 and j-1>=0 and visit[i-1][j-1]==0 and mat[i-1][j-1]!=0):
        return (i-1,j-1)
        

    if (i-1>=0 and j+1<c and visit[i-1][j+1]==0 and mat[i-1][j+1]!=0):
        return (i-1,j+1)

    if (i+1<r and j-1>=0 and visit[i+1][j-1]==0 and mat[i+1][j-1]!=0):
        return (i+1,j-1)
    if (i+1<r and j+1<c and visit[i+1][j+1]==0 and mat[i+1][j+1]!=0):
        return (i+1,j+1)

    return (-1,-1)



def start_point(img,r,c):
    im = img.copy()

    im[im==255] = 1
    dummy = cv2.cvtColor(img.copy(),cv2.COLOR_GRAY2BGR)
    #cv2.imshow("start_point",dummy)
    for i in range(1,r-1):
        for j in range(1,c-1):
            if im[i][j]==1:
                roi = im[i-1:i+2, j-1:j+2]
                p = np.sum(roi)
                if(p==2):
                    dummy[i][j] = [0,0,255]
                    return (i,j)

def order_points(img):
    global r,c,visit
    r,c = img.shape
    visit = np.zeros((r,c))
    x,y = start_point(img,r,c)

    flag = True
    points = [[x,y]]

    while(flag):
        x,y = dfs(img,x,y)
        if x!=-1 and y!=-1:
            points.append([x,y])
        else:
            flag=False

    return points

def getinflections(img,points):
    kernels = [np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32),
                np.array([[0,1,0],[0,1,0],[0,1,0]],dtype=np.float32),
                np.array([[0,0,1],[0,1,0],[1,0,0]],dtype=np.float32),
                np.array([[0,0,0],[1,1,1],[0,0,0]],dtype=np.float32)]

    pts = []
    img[img==255] = 1

    for i,j in points:
        roi = img[i-1:i+2, j-1:j+2]
        flag = 0
        for k in kernels:
            p = np.sum(k)
            r = np.sum(np.multiply(roi,k))
            if(r==p):
                flag=1
                break

        if flag==0:
            pts.append([i,j])

    return pts


def compute_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    # if(angle == 0 or str(angle) == 'nan'):
    #     print('found : '+str(angle))
    #     print(a,b,c)
    #     print()

    return np.degrees(angle)

def get_angles(inflects):
    if(len(inflects)<3):
        return [180]

    n = len(inflects)
    angles = []
    for i in range(n-2):
        angles.append(compute_angle(inflects[i],inflects[i+1],inflects[i+2]))

    return angles

def contour_inflections(img):

    cnts = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)[-2]

    #temp = np.ones(img.shape, np.uint8) * 255
    #cv2.drawContours(temp, cnts, -1, (0, 255, 0), 1)
    #cv2.imshow("contor_inflections_img",temp)
    #cv2.imshow("img_w",img)

    cnt = sorted(cnts, key=cv2.contourArea)[-1]  #对所有的cnts进行排序

    #temp = np.ones(img.shape, np.uint8) * 255
    #cv2.drawContours(temp, cnt, -1, (0, 255, 0), 1)
    #cv2.imshow("contor_inflections_img",temp)

    arclen = cv2.arcLength(cnt, True) #计算轮廓的弧长

    epsilon = arclen * 0.0075

    approx = cv2.approxPolyDP(cnt, epsilon, False) #将连续的点进行折线化
    #canvas = cv2.cvtColor(np.zeros(img.shape,np.uint8),cv2.COLOR_GRAY2BGR)
    #cv2.drawContours(canvas, [approx], -1, (0,0,255), 1)
    #cv2.imshow("canvas",canvas)
    pts = []
    for pt in approx:
        i,j = pt[0][1], pt[0][0]
        if([i,j] not in pts):
            pts.append([i,j])
            # canvas[i][j] = [0,255,0]
    
    # cv2.imshow('smooth',canvas)
    # cv2.waitKey(0)
    
    return pts