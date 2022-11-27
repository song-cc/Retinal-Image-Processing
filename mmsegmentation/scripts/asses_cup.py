import os

import SimpleITK as sitk
import cv2
import numpy as np
import matplotlib.pyplot as plt

def dice(lP, lT):
    quality = dict()
    labelPred = sitk.GetImageFromArray(lP, isVector=False)
    labelTrue = sitk.GetImageFromArray(lT, isVector=False)

    dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue > 0.5, labelPred > 0.5)

    quality["dice"] = dicecomputer.GetDiceCoefficient()

    return quality["dice"]


if __name__ == "__main__":
    #label_path = "/media/songcc/data/songcc/Retinal/dataset/Cycle_GAN_data/small_cycle_gt"
    # label_path = "/home/songcc/code/mmseg/mmseg/data/IDRi/IDRi_gt"
    # label_path = "/media/songcc/data/songcc/Retinal/dataset/Drishiti-GS1_op/DR_cupGT"
    # label_path = "/media/songcc/data/songcc/Retinal/dataset/MICCAI2021/GAMMA_cupGT"
    #label_path = "//home/songcc/code/mmseg/mmseg/data/REFUGE/Validation400_optic"
    label_path = "/media/songcc/data/songcc/Retinal/dataset/REFUGE/Test_cupGT"
    #label_path = "/home/songcc/code/mmseg/mmseg/data/REFUGE/Train400_optic"
    # pre_path = "/media/songcc/data/songcc/Retinal/code/mmsegmentation/work_dirs/fcn_cup_Image/MICCAI"
    pre_path = "/media/songcc/data/songcc/Retinal/optic_segmentation/Seg_project/data/result"
    names = os.listdir(label_path)
    dice_res = []
    for name in names:
        label = cv2.imread(os.path.join(label_path,name))
        label = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
        #print(os.path.join(pre_path,name[:-4] + ".png"))
        pre = cv2.imread(os.path.join(pre_path,name[:-4] + ".png"))
        pre = cv2.cvtColor(pre,cv2.COLOR_BGR2GRAY)

        _,label = cv2.threshold(label,10,255,cv2.THRESH_BINARY)
        _,pre = cv2.threshold(pre,10,255,cv2.THRESH_BINARY)

        dc = dice(label,pre)
        print(dc,name)
        dice_res.append(dc)

    dice_res = np.array(dice_res)
    print(np.min(dice_res),"min_dice")
    print(np.mean(dice_res),"avg_dice")
    print(np.max(dice_res),"max_dice")