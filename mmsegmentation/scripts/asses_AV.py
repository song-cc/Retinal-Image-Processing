import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, mean_squared_error
import numpy as np
import os
import cv2
import ipdb

def misc_measures(true_vessel_arr, pred_vessel_arr):
    cm=confusion_matrix(true_vessel_arr, pred_vessel_arr)
    mse = mean_squared_error(true_vessel_arr, pred_vessel_arr)

    try:
        acc=1.*(cm[0,0]+cm[1,1])/np.sum(cm)
        sensitivity=1.*cm[1,1]/(cm[1,0]+cm[1,1])
        specificity=1.*cm[0,0]/(cm[0,1]+cm[0,0])
        precision=1.*cm[1,1]/(cm[1,1]+cm[0,1])
        G = np.sqrt(sensitivity*specificity)
        F1_score_2 = 2*precision*sensitivity/(precision+sensitivity)
        iou = 1.*cm[1,1]/(cm[1,0]+cm[0,1]+cm[1,1])
        return acc, sensitivity, specificity, precision, G, F1_score_2, mse, iou
    
    except:

        return 0,0,0,0,0,0,0,0

if __name__ == "__main__":
    pred_path = "/media/songcc/data/songcc/Retinal/code/mmsegmentation/work_dirs/drive_AV"
    # pred_path = "/media/songcc/data/songcc/Retinal/Learning-AVSegmentation-main/DRIVE_AV/Final_pre/res_got_eval"
    gt_path = "/media/songcc/data/songcc/Retinal/AVSegmentation_data/DRIVE_AV/test/gt_eval"

    res = []
    names = os.listdir(pred_path)
    for name in names:
        pred = cv2.imread(os.path.join(pred_path,name))
        h,w,_ = np.shape(pred)
        gt = cv2.imread(os.path.join(gt_path,name))
        pred_class = np.zeros([h,w])
        gt_class = np.zeros([h,w])
        # ipdb.set_trace()
        pred_class[pred[:,:,0]==255] = 1
        pred_class[pred[:,:,2]==255] = 2

        gt_class[gt[:,:,0]==255] = 1
        gt_class[gt[:,:,2]==255] = 2
        pred_class = pred_class.flatten()
        gt_class = gt_class.flatten()

        res.append(misc_measures(gt_class,pred_class))
        #acc, sensitivity, specificity, precision, G, F1_score_2, mse, iou = 0
    res = np.array(res)
    print("acc, sensitivity, specificity, precision, G, F1_score_2, mse, iou")
    print(np.mean(res,axis=0))