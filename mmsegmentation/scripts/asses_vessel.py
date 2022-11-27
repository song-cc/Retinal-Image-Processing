from operator import attrgetter
import os
import cv2
import numpy as np
from skimage import morphology, graph,measure
from random import randint
import ipdb

def topo_metric(gt, pred, thresh, n_paths):

    # 0, 1 and 2 mean, respectively, that path is infeasible, shorter/larger and correct
    result = []

    # binarize pred according to thresh
    pred_bw = (pred>thresh).astype(int)
    pred_cc = morphology.label(pred_bw)

    # get centerlines of gt and pred
    gt_cent = morphology.skeletonize(gt>0.5)
    gt_cent_cc = morphology.label(gt_cent)
    pred_cent = morphology.skeletonize(pred_bw)
    pred_cent_cc = morphology.label(pred_cent)

    # costs matrices
    gt_cost = np.ones(gt_cent.shape)
    gt_cost[gt_cent==0] = 10000
    pred_cost = np.ones(pred_cent.shape)
    pred_cost[pred_cent==0] = 10000
    
    # build graph and find shortest paths
    for i in range(n_paths):

        # pick randomly a first point in the centerline
        R_gt_cent, C_gt_cent = np.where(gt_cent==1)
        idx1 = randint(0, len(R_gt_cent)-1)
        label = gt_cent_cc[R_gt_cent[idx1], C_gt_cent[idx1]]
        ptx1 = (R_gt_cent[idx1], C_gt_cent[idx1])

        # pick a second point that is connected to the first one
        R_gt_cent_label, C_gt_cent_label = np.where(gt_cent_cc==label)
        idx2 = randint(0, len(R_gt_cent_label)-1)
        ptx2 = (R_gt_cent_label[idx2], C_gt_cent_label[idx2])

        # if points have different labels in pred image, no path is feasible
        if (pred_cc[ptx1] != pred_cc[ptx2]) or pred_cc[ptx1]==0:
            result.append(0)

        else:
            # find corresponding centerline points in pred centerlines
            R_pred_cent, C_pred_cent = np.where(pred_cent==1)
            poss_corr = np.zeros((len(R_pred_cent),2))
            poss_corr[:,0] = R_pred_cent
            poss_corr[:,1] = C_pred_cent
            poss_corr = np.transpose(np.asarray([R_pred_cent, C_pred_cent]))
            dist2_ptx1 = np.sum((poss_corr-np.asarray(ptx1))**2, axis=1)
            dist2_ptx2 = np.sum((poss_corr-np.asarray(ptx2))**2, axis=1)
            corr1 = poss_corr[np.argmin(dist2_ptx1)]
            corr2 = poss_corr[np.argmin(dist2_ptx2)]
            
            # find shortest path in gt and pred
            
            gt_path, cost1 = graph.route_through_array(gt_cost, ptx1, ptx2)
            gt_path = np.asarray(gt_path)

            pred_path, cost2 = graph.route_through_array(pred_cost, corr1, corr2)
            pred_path = np.asarray(pred_path)


            # compare paths length
            path_gt_length = np.sum(np.sqrt(np.sum(np.diff(gt_path, axis=0)**2, axis=1)))
            path_pred_length = np.sum(np.sqrt(np.sum(np.diff(pred_path, axis=0)**2, axis=1)))
            if pred_path.shape[0]<2:
                result.append(2)
            else:
                if ((path_gt_length / path_pred_length) < 0.9) or ((path_gt_length / path_pred_length) > 1.1):
                    result.append(1)
                else:
                    result.append(2)

    return result.count(0), result.count(1), result.count(2)

def connect_analysis(pred):
    all = np.sum(pred)
    _,pred = cv2.threshold(pred,0.5,255,cv2.THRESH_BINARY)
    h,w = np.shape(pred)

    label = measure.label(pred)
    region_props = measure.regionprops(label)
    maxNumPixel = max(region_props,key=attrgetter("area"))
    maxNumPixel = maxNumPixel.area * 0.03
    N = len(region_props)
    right = 0
    sums = 0

    for i in range(N):
        area = region_props[i].area
        if area >= maxNumPixel:
            right += 1
            sums += area
    
    return right / N, sums / all


def ACC(gt,pre):
    length,width = np.shape(gt)
    all = length * width
    right = all - np.sum(np.abs(gt - pre))
    return right / all

def SE(gt,pre):
    # middle = gt + pre
    # res = np.zeros(np.shape(middle))
    # res[middle>=2] = 1
    TP = np.sum(np.multiply(gt,pre))
    
    P_all = np.sum(gt)
    return TP / P_all

def SP(gt,pre):
    length,width = np.shape(gt)
    middle = gt + pre
    res = np.zeros(np.shape(middle))
    res[middle <= 0] = 1
    TN = np.sum(res)
    N_all = length * width - np.sum(gt)

    return TN / N_all

if __name__ == "__main__":

    pre_path = "/media/songcc/data/songcc/Retinal/DRIVE/test/SegFormer_res"
    label_path = "/media/songcc/data/songcc/Retinal/DRIVE/test/1st_png"
    names = os.listdir(pre_path)
    
    sp = []
    se = []
    acc = []
    connect_rate = []
    max_connect = []

    for name in names:
        gt = cv2.imread(os.path.join(label_path,name))
        pre = cv2.imread(os.path.join(pre_path,name))
        gt = cv2.cvtColor(gt,cv2.COLOR_BGR2GRAY) // 255
        pre = cv2.cvtColor(pre,cv2.COLOR_BGR2GRAY) // 255
        gt = gt.astype(np.int16)
        pre = pre.astype(np.int16)
        per_acc = ACC(gt,pre)
        per_sp = SP(gt,pre)
        per_se = SE(gt,pre)
        # a,b,c = topo_metric(gt, pre, 0.5, 100)
        con_rate,max_rate = connect_analysis(pre)
        
        acc.append(per_acc)
        se.append(per_se)
        sp.append(per_sp)
        connect_rate.append(con_rate)
        max_connect.append(max_rate)
        
        print(name,"Acc:",per_acc,"SE:",per_se,"SP:",per_sp,"connect_rate:",con_rate,"maxConnect_rate:",max_rate)
    
    print("=============================================================")
    print("Acc:",np.mean(acc),"SE:",np.mean(se),"SP:",np.mean(sp),"connect_rate:",np.mean(connect_rate),"maxConnect_rate:",np.mean(max_connect))


