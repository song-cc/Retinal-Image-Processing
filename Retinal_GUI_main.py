import numpy as np
import sys
import os
from PyQt5.QtCore import pyqtSignal
import pandas as pd

sys.path.append("/home/songcc/code/Retial-GUI/mmsegmentation")
from Retail_GUI import Ui_Form,Ui_load_result
import cv2
import torch
from PyQt5.QtWidgets import QApplication,QFileDialog,QDialog,QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap,QImage,QGuiApplication
from image_process import load_process,got_Roi_image,got_connect_bbox
from mmsegmentation.scripts.post_process_AV import got_AV
from mmsegmentation.scripts.post_process import connect_analysis
from loadModel import Stage1_SegMentor,Roi_Detector,Stage2_SegMentor,sa_UNet,mmsegmentor
from measurement import *
from multiprocessing import Pool
import ipdb
device = torch.device("cuda:0")
import torch

class load_result_GUI(QDialog,Ui_load_result):
    finised_signal = pyqtSignal(str)
    def __init__(self, parent = None) -> None:
        super(load_result_GUI,self).__init__(parent)
        self.setupUi(self)
        self.load_OD.clicked.connect(self.load_OD_clicked)
        self.load_OC.clicked.connect(self.load_OC_clicked)
        self.load_vessel.clicked.connect(self.load_vessel_clicked)
        self.load_AV.clicked.connect(self.load_AV_clicked)
        self.finised_signal.connect(parent.gotImgFromPreSeg)
    
    def load_OD_clicked(self):
        self.load_img()
        self.close()
        self.finised_signal.emit("OD")

    def load_OC_clicked(self):
        self.load_img()
        self.close()
        self.finised_signal.emit("OC")

    def load_vessel_clicked(self):
        self.load_img()
        self.close()
        self.finised_signal.emit("vessel")

    def load_AV_clicked(self):
        self.load_img()
        self.close()
        self.finised_signal.emit("AV")
    
    def load_img(self):
        imgName, _ = QFileDialog.getOpenFileName(self, "图像加载", filter="Image Files (*.png)")
        if imgName == "":
            return
        self.img = cv2.imread(imgName)
    
    def got_img(self):
        return self.img

class GUI(QWidget,Ui_Form):
    def __init__(self,parent=None):
        super(GUI, self).__init__(parent)
        self.setupUi(self)
        
        self.OC_show = True
        self.OD_show = True
        self.vessel_show = True
        self.AV_show = True
        self.OD_diametr_show = True

        self.open_file.clicked.connect(self.slot_open_file_clicked)
        self.image_view.erro_signal.connect(self.erro_display)
        self.closed.clicked.connect(self.slot_closed_clicked)
        self.biggest.clicked.connect(self.slot_biggest_clicked)
        self.smallest.clicked.connect(self.slot_smallest_clicked)
        self.OPD_segment.clicked.connect(self.slot_OPD_segment_clicked)
        self.OPC_segment.clicked.connect(self.slot_OPC_segment_clicked)
        self.load_model.pressed.connect(self.load_model_cliced)
        self.vessel_segment.clicked.connect(self.vessel_segment_clicked)
        self.AV_segment.clicked.connect(self.AV_segment_clicked)
        self.oDandoC_diameter.clicked.connect(self.oDandoC_diameter_clicked)
        self.diameter_estimate.clicked.connect(self.diameter_estimate_clicked)
        self.CRAEandCRVE.clicked.connect(self.CRAEandCRVE_clicked)
        self.branch_angle.clicked.connect(self.branch_angle_clicked)
        self.curvature.clicked.connect(self.curvature_clicked)
        self.fracta_dim.clicked.connect(self.fracta_dim_clicked)
        self.save_result.clicked.connect(self.save_result_clicked)
        self.load_result.clicked.connect(self.load_result_clicked)

    def slot_closed_clicked(self):
        self.close()

    def slot_smallest_clicked(self):
        self.showMinimized()

    def slot_biggest_clicked(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def slot_open_file_clicked(self):
        self.image_view.clear()
        self.orgementaion_view.clear()
        self.Display_info.clear()
        self.Display_para.clear()
        imgName, filetype = QFileDialog.getOpenFileName(self,
                                                        "图像加载",
                                                        filter="All Files (*);;Text Files (*.txt)")
        if imgName == "":
            return
        
        self.Display_info.append(imgName + ' is processing')
        self.Display_info.setAlignment(Qt.AlignTop)
        img = cv2.imread(imgName)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        w_win,h_win = self.image_view.width(),self.image_view.height()
        img = load_process(img,w_win,h_win)
        self.image_view.cur_img = img
        h,w,c = np.shape(img)
        img = QImage(img.data,w,h,w * 3,QImage.Format_RGB888)

        self.image_view.x_0 = self.image_view.geometry().left()
        self.image_view.y_0 = self.image_view.geometry().top()
        self.image_view.h = self.image_view.geometry().height()
        self.image_view.w = self.image_view.geometry().width()
        self.image_view.res = np.zeros((w,h)).astype(np.uint8)

        self.image_view.setPixmap(QPixmap.fromImage(img))
        
        self.result = {}
        split_ele = "/" if "/" in imgName else "\\"
        name = imgName.split(split_ele)[-1]
        self.result["Name"] = name.split(".")[0]
        self.result["img"] = self.image_view.cur_img

    def save_result_clicked(self):
        filepath  = QFileDialog.getExistingDirectory(self,"保存结果")        
        if not "result" in dir(self):
            return
        keys = self.result.keys()
        result_dataFrame = {}
        img_keys = ["img","OD","OC","vessel","AV"]
        dataFrame_keys = ["Name","OC_diameter","OD_diameter","OD_center","CDR","CRAE","CRVE",
        "artery branching angle","vein branching angle","artery torts","artery arcbased","artery arc_torts",
        "vein torts","vein arcbased","vein arc_torts","artery fractal dimension","vein fractal dimension"]
        
        for key in keys:
            if key in img_keys:
                try:
                    cv2.imwrite(os.path.join(filepath,self.result["Name"] + key + ".png"),self.result[key][:,:,::-1])
                except:
                    cv2.imwrite(os.path.join(filepath,self.result["Name"] + key + ".png"),self.result[key])
            elif key in dataFrame_keys:
                save_key = key.replace(" ","_")
                result_dataFrame[save_key] = [self.result[key]]
        result_dataFrame = pd.DataFrame(result_dataFrame)
        result_dataFrame.to_excel(os.path.join(filepath,"res.xlsx"),index=False)

    def load_model_cliced(self):
        text = "loading Stage1 Segmentor ...  "
        self.Display_info.append(text)
        QGuiApplication.processEvents()
        self.course_Segmentor = Stage1_SegMentor(device=device,
                                            model_path="Stage1/Stage1.params")
        text = "load Stage1 Segmentor finished  "
        self.Display_info.append(text)
        QGuiApplication.processEvents()


        text = "loading ROI Detector ...  "
        self.Display_info.append(text)
        QGuiApplication.processEvents()
        self.roi_Detector = Roi_Detector(device=device,model_path="ROI_Detection/ROI_Detector.param")
        text = "load ROI Detector finished "
        self.Display_info.append(text)
        QGuiApplication.processEvents()


        self.Display_info.append("loading OD Stage2 Segmentor ....")
        QGuiApplication.processEvents()
        self.fine_Segmentor = Stage2_SegMentor(device=device,model_path="Stage2/CE-Net.th")
        text = "load OD Stage2 Segmentor finished "
        self.Display_info.append(text)
        QGuiApplication.processEvents()


        self.Display_info.append("loading OC Stage2 Segmentor ....")
        self.fine_cupSegmentor = Stage2_SegMentor(device=device,model_path="Stage2/W-Net-cup.pth")
        text = "load OC Stage2 Segmentor finished "
        self.Display_info.append(text)
        QGuiApplication.processEvents()

        self.Display_info.append("loading course vessel Segmentor1")
        self.courseVessel_model_1 = sa_UNet(model_path="SAUNet/Model/SA_UNet.h5")
        text = "load course vessel Segmentor finished "
        self.Display_info.append(text)
        QGuiApplication.processEvents()

        self.Display_info.append("loading course vessel SegMentor2")
        self.courseVessel_model_2 = mmsegmentor("mmsegmentation/work_dirs/segformer_mit-b4_512x512_160k_drive/segformer_mit-b4_512x512_160k_drive.py",
                                    "mmsegmentation/work_dirs/segformer_mit-b4_512x512_160k_drive/latest.pth")
        text = "load course vessel Segmentor finished"
        self.Display_info.append(text)
        QGuiApplication.processEvents()

        self.Display_info.append("loading fine vessel SegMentor")
        self.fineVessel_model = mmsegmentor("mmsegmentation/work_dirs/segformer_mit-b4_512x512_160k_driveMulti/segformer_mit-b4_512x512_160k_driveMulti.py",
                                    "mmsegmentation/work_dirs/segformer_mit-b4_512x512_160k_driveMulti/latest.pth")
        text = "load fine vessel Segmentor finished"
        self.Display_info.append(text)
        QGuiApplication.processEvents()

        self.Display_info.append("loading AV_vessel Segmentor")
        self.AV_model = mmsegmentor("mmsegmentation/work_dirs/segformer_mit-b4_512x512_160k_AVdrive_Original/segformer_mit-b4_512x512_160k_AVdrive.py",
                                    "mmsegmentation/work_dirs/segformer_mit-b4_512x512_160k_AVdrive_Original/latest.pth")
        text = "load AV_vessel Segmentor finished "
        self.Display_info.append(text)
        QGuiApplication.processEvents()

    def erro_display(self,val):
        self.Display_info.append(val)

    def slot_OPD_segment_clicked(self):
        if self.OD_show and "OD" in self.result.keys() and not self.result["OD"] is None:
            self.orgementaion_view.clear()
            img = QImage(self.result["OD"], self.result["w"], self.result["h"], self.result["w"], QImage.Format_Grayscale8)
            self.orgementaion_view.setPixmap(QPixmap.fromImage(img).scaled(self.orgementaion_view.height(),self.orgementaion_view.width()))
            return
        img = self.image_view.cur_img
        try:
            w,h,_ = np.shape(img)
            img = cv2.resize(img,dsize=(256,256))
        except:
            self.Display_info.append("You must load RGB image with (w,h,3) before do this step")
            return

        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.unsqueeze(0)

        try:
            res = self.course_Segmentor(img_tensor,(w,h))
        except:
            self.Display_info.append("You must load model before do this step")
            return

        roiS,bboxes = got_Roi_image(self.image_view.cur_img,res)
        
        try:
            roi_index = self.roi_Detector(roiS)
            self.result["roi"] = roiS[roi_index]
            self.result["bboxes"] = bboxes[roi_index]
            self.result["h"] = h
            self.result["w"] = w
        except:
            self.Display_info.append("You must load model before do this step")
            return

        try:
            res_OD = self.fine_Segmentor(roiS[roi_index])
        except:
            self.Display_info.append("You must load model before do this step")
            return
        OD_show = np.zeros([w,h])
        OD_show = OD_show.astype(np.uint8)
        
        OD_show[bboxes[roi_index][0]:bboxes[roi_index][1],bboxes[roi_index][2]:bboxes[roi_index][3]] = res_OD
        self.result["OD"] = OD_show

        if self.OD_show:
            self.orgementaion_view.clear()
            img = QImage(OD_show, w, h, w, QImage.Format_Grayscale8)
            self.orgementaion_view.setPixmap(QPixmap.fromImage(img).scaled(self.orgementaion_view.height(),self.orgementaion_view.width()))
        
    def slot_OPC_segment_clicked(self):
        if self.OC_show and "OC" in self.result.keys() and not self.result["OC"] is None:
            self.orgementaion_view.clear()
            img = QImage(self.result["OC"],self.result["w"], self.result["h"], self.result["w"], QImage.Format_Grayscale8)
            self.orgementaion_view.setPixmap(QPixmap.fromImage(img).scaled(self.orgementaion_view.height(),self.orgementaion_view.width()))
            return
        
        if not "roi" in self.result.keys() or self.result["roi"] is None:
            img = self.image_view.cur_img
            try:
                w,h,_ = np.shape(img)
                img = cv2.resize(img,dsize=(256,256))
            except:
                self.Display_info.append("You must load RGB image with (w,h,3) before do this step")
                return

            img_tensor = torch.from_numpy(img)
            img_tensor = img_tensor.unsqueeze(0)

            try:
                res = self.course_Segmentor(img_tensor,(w,h))
            except:
                self.Display_info.append("You must load model before do this step")
                return

            roiS,bboxes = got_Roi_image(self.image_view.cur_img,res)

            try:
                roi_index = self.roi_Detector(roiS)
                self.result["roi"] = roiS[roi_index]
                self.result["bboxes"] = bboxes[roi_index]
                self.result["w"] = w
                self.result["h"] = h
            except:
                self.Display_info.append("You must load model before do this step")
                return

        try:
            res_OC = self.fine_cupSegmentor(self.result["roi"])
        except:
            self.Display_info.append("You must load model before do this step")
            return
        OC_show = np.zeros([self.result["w"],self.result["h"]])
        OC_show = OC_show.astype(np.uint8)

        OC_show[self.result["bboxes"][0]:self.result["bboxes"][1],self.result["bboxes"][2]:self.result["bboxes"][3]] = res_OC
        self.result["OC"] = OC_show

        if self.OC_show:
            img = QImage(OC_show, self.result["w"],self.result["h"],self.result["w"], QImage.Format_Grayscale8)
            self.orgementaion_view.clear()
            self.orgementaion_view.setPixmap(QPixmap.fromImage(img).scaled(self.orgementaion_view.height(),self.orgementaion_view.width()))

    def vessel_segment_clicked(self):
        if self.vessel_show and "vessel" in self.result.keys() and self.result["vessel"] is not None:
            self.orgementaion_view.clear()
            img = QImage(self.result["vessel"], self.orgementaion_view.width(), self.orgementaion_view.height(),self.orgementaion_view.width(), QImage.Format_Grayscale8)
            self.orgementaion_view.setPixmap(QPixmap.fromImage(img).scaled(self.orgementaion_view.height(),self.orgementaion_view.width()))
            return 
        img0 = self.image_view.cur_img
        img0 = img0[:,:,::-1]
        img = cv2.resize(img0,(592,592))
        img = img[np.newaxis,...]
        img = img.astype('float32') / 255.
        courseVessel = self.courseVessel_model_1.model.predict(img)
        y = courseVessel.reshape((592,592))
        y = cv2.resize(y,(self.orgementaion_view.width(),self.orgementaion_view.width()),interpolation=cv2.INTER_LINEAR)
        courseVessel_1 = np.zeros((self.orgementaion_view.width(),self.orgementaion_view.width())).astype(np.uint8)
        courseVessel_1[y >= 0.5] = 255
        courseVessel_1[y < 0.5] = 0
        self.result["courseVessel"] = courseVessel_1

        palette = np.array([[0],[255]])
        courseVessel_2 = self.courseVessel_model_2(img0,palette)
        self.result["courseVesse2"] = courseVessel_2
        
        img_use = np.zeros((self.orgementaion_view.width(),self.orgementaion_view.width(),5))
        img_use[:,:,:3] = img0
        img_use[:,:,3] = courseVessel_1
        img_use[:,:,4] = courseVessel_2
        vessel = self.fineVessel_model(img_use,palette)
        vessel = connect_analysis(vessel)
        self.result["vessel"] = vessel
        if self.vessel_show:
            self.orgementaion_view.clear()
            img = QImage(vessel, self.orgementaion_view.width(), self.orgementaion_view.height(),self.orgementaion_view.width(), QImage.Format_Grayscale8)
            self.orgementaion_view.setPixmap(QPixmap.fromImage(img).scaled(self.orgementaion_view.height(),self.orgementaion_view.width()))
        
    def AV_segment_clicked(self):
        if self.AV_show and "AV" in self.result.keys() and not self.result["AV"] is None:
            self.orgementaion_view.clear()
            img = QImage(self.result["AV"], self.orgementaion_view.width(), self.orgementaion_view.height(),self.orgementaion_view.width() * 3, QImage.Format_RGB888)
            self.orgementaion_view.setPixmap(QPixmap.fromImage(img))
            return 
        
        if "vessel" not in self.result.keys() or self.result["vessel"] is None:
            self.vessel_show = False
            self.vessel_segment_clicked()
            self.vessel_show = True
        palette = np.array([[ 0,   0,   0],
                    [ 0,   0, 255],
                    [255,   0,   0]])
        res = self.AV_model(self.image_view.cur_img[:,:,::-1],palette)
        vein = np.zeros((self.image_view.height(),self.image_view.width())).astype(np.uint8)
        vein[res[:,:,0]==255] = 255
        res = got_AV(self.result["vessel"],vein)
        self.result["AV"] = res
        if self.AV_show:
            self.orgementaion_view.clear()
            img = QImage(res, self.orgementaion_view.width(), self.orgementaion_view.height(),self.orgementaion_view.width() * 3, QImage.Format_RGB888)
            self.orgementaion_view.setPixmap(QPixmap.fromImage(img))   
        
    def oDandoC_diameter_clicked(self):
        
        if "OD" not in self.result.keys() or self.result["OD"] is None:
            self.OD_show = False
            self.slot_OPD_segment_clicked()
            self.OD_show = True
        OD_diameter,center_OD = got_OD_diameter(self.result["OD"])
        if "OC" not in self.result.keys() or self.result["OC"] is None:
            self.OC_show = False
            self.slot_OPC_segment_clicked()
            self.OC_show = True
        OC_diameter,_ = got_OD_diameter(self.result["OC"])
        self.result["OC_diameter"] = OC_diameter
        self.result["OD_diameter"] = OD_diameter
        self.result["OD_center"] = center_OD
        CDR = OC_diameter / OD_diameter if OD_diameter > 0 else -1
        self.result["CDR"] = CDR
        
        if self.OD_diametr_show:
            self.Display_para.append("OC diameter is :" + str(round(OC_diameter,2)))
            self.Display_para.append("OD diameter is :" + str(round(OD_diameter,2)))
            self.Display_para.append("CDR diameter is :" + str(round(CDR,2)))
        
    def diameter_estimate_clicked(self):
        if "vessel" not in self.result.keys() or self.result["vessel"] is None:
            self.vessel_show = False
            self.vessel_segment_clicked()
            self.vessel_show = True
        self.result["vessel_seg"] = got_vessel_seg(self.result["vessel"]) * 255
        self.result["vessel_seg"] = self.result["vessel_seg"].astype(np.uint8)
        self.orgementaion_view.clear()
        img = QImage(self.result["vessel_seg"], self.orgementaion_view.width(), self.orgementaion_view.height(),self.orgementaion_view.width(), QImage.Format_Grayscale8)
        self.orgementaion_view.setPixmap(QPixmap.fromImage(img).scaled(self.orgementaion_view.height(),self.orgementaion_view.width()))
        self.orgementaion_view.call = True
        
    def after_diameter_estimate_clicked(self,env):
        x,y = env
        vessel_seg = region_grow(self.result["vessel_seg"],[x,y])
        bbox = got_connect_bbox(vessel_seg)
        bbox = bbox[0]
        vessel_seg = vessel_seg[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        vessel_seg = vessel_seg / 255
        vessel_seg = vessel_seg.astype(np.uint8)
        thinVessel = morphology.skeletonize(vessel_seg)
        
        vessel_width = got_one_width(vessel_seg * 255,thinVessel)
        self.Display_para.append(" the vessel you selected is " + str(round(vessel_width,2)) + " pixel")

    def CRAEandCRVE_clicked(self):
        if "AV" not in self.result.keys() or self.result["AV"] is None:
            self.AV_show = False
            self.AV_segment_clicked()
            self.AV_show = True
        
        if "OD_diamater" not in self.result.keys() or self.result["OD_diamater"] is None:
            self.OD_diametr_show = False
            self.oDandoC_diameter_clicked()
            self.OD_diametr_show = True

        OD_R = self.result["OD_diameter"] / 2
        OD_center = self.result["OD_center"] 
        roi_img = np.zeros_like(self.result["OD"])

        w,h = np.shape(roi_img)

        for x in range(w):
            for y in range(h):
                S = ((x - OD_center[1]) ** 2 + (y - OD_center[0]) ** 2) ** 0.5
                if 1.5 * OD_R <= S <= 2 * OD_R:
                    roi_img[x,y] = 1
        
        artery = self.result["AV"][:,:,0]
        vein = self.result["AV"][:,:,2]
        
        artery = np.multiply(artery,roi_img)
        vein = np.multiply(vein,roi_img)
        
        arteries = got_every_vessel(artery)
        veins = got_every_vessel(vein)
        
        artery_widthes = self.got_widthes(arteries)
        vein_widthes = self.got_widthes(veins)
        
        CRAE,CRVE = got_CRAEandCRVE(vein_widthes,artery_widthes,self.result["OD_diameter"])
        self.result["CRAE"] = CRAE
        self.result["CRVE"] = CRVE
        self.Display_para.append("CRAE is " + str(round(CRAE,2)))
        self.Display_para.append("CRVE is " + str(round(CRVE,2)))

    def got_widthes(self,vessels):
        res = []
        for vessel_seg in vessels:
            if np.max(vessel_seg) == 255:
                vessel_seg = vessel_seg / 255
            vessel_seg = vessel_seg.astype(np.uint8)
            thinVessel = morphology.skeletonize(vessel_seg)

            y_target,_ = np.nonzero(thinVessel)
            if np.shape(y_target)[0] < 10:
                continue
            res.append(got_one_width(vessel_seg * 255,thinVessel))
        
        return res

    def branch_angle_clicked(self):
        if "AV" not in self.result.keys() or self.result["AV"] is None:
            self.AV_show = False
            self.AV_segment_clicked()
            self.AV_show = True

        artery = self.result["AV"][:,:,0]
        vein = self.result["AV"][:,:,2]
        
        artery_branching_angle = got_angle_reference(artery)
        vein_branching_angle = got_angle_reference(vein)
        
        self.result["artery branching angle"] = artery_branching_angle
        self.result["vein branching angle"] = vein_branching_angle
        self.Display_para.append("artery branching angle : " + str(round(artery_branching_angle,2)))
        self.Display_para.append("vein branching angle : " + str(round(vein_branching_angle,2)))
    
    def curvature_clicked(self):
        if "AV" not in self.result.keys() or self.result["AV"] is None:
            self.AV_show = False
            self.AV_segment_clicked()
            self.AV_show = True

        artery = self.result["AV"][:,:,0]
        vein = self.result["AV"][:,:,2]
        
        artery_torts = got_curvature(artery)
        vein_torts = got_curvature(vein)
        
        self.result["artery torts"] = artery_torts[0]
        self.result["artery arcbased"] = artery_torts[1]
        self.result["artery arc_torts"] = artery_torts[2]
        self.result["vein torts"] = vein_torts[0]
        self.result["vein arcbased"] = vein_torts[1]
        self.result["vein arc_torts"] = vein_torts[2]
        self.Display_para.append("artery curvature : ") 
        self.Display_para.append("        torts:" + str(round(artery_torts[0],2)))
        self.Display_para.append("        arcbased" + str(round(artery_torts[1],2)))
        self.Display_para.append("        arc_torts" + str(round(artery_torts[2],2)))
        self.Display_para.append("vein curvature : ")
        self.Display_para.append("        torts:" + str(round(vein_torts[0],2)))
        self.Display_para.append("        arcbased" + str(round(vein_torts[1],2)))
        self.Display_para.append("        arc_torts" + str(round(vein_torts[2],2)))
    
    def fracta_dim_clicked(self):
        if "AV" not in self.result.keys() or self.result["AV"] is None:
            self.AV_show = False
            self.AV_segment_clicked()
            self.AV_show = True

        artery = self.result["AV"][:,:,0]
        vein = self.result["AV"][:,:,2]

        artery = cv2.resize(artery,(256,256),interpolation=cv2.INTER_NEAREST)
        vein = cv2.resize(vein,(256,256),interpolation=cv2.INTER_NEAREST)

        with Pool(2) as p:
            outputs = p.map(got_fractal_dimension,[artery,vein])

        artery_fd = outputs[0]
        vein_fd = outputs[1]

        self.result["artery fractal dimension"] = artery_fd
        self.result["vein fractal dimension"] = vein_fd
        self.Display_para.append("artery fractal dimension is :" + str(round(artery_fd,2)))
        self.Display_para.append("vein fractal dimension is :" + str(round(vein_fd,2)))
    
    def load_result_clicked(self):
        self.load_result_window = load_result_GUI(self)
        self.load_result_window.exec_()

    def gotImgFromPreSeg(self,env):
        if "result" not in dir(self):
            self.Display_info.append("you must load CFP image first")
            return 

        if env == "OD" or env == "OC" or env == "vessel":
            self.result[env] = cv2.cvtColor(self.load_result_window.got_img(),cv2.COLOR_BGR2GRAY)
            self.orgementaion_view.clear()
            img = QImage(self.result[env], self.orgementaion_view.height(),self.orgementaion_view.width(), self.orgementaion_view.height(), QImage.Format_Grayscale8)
            self.orgementaion_view.setPixmap(QPixmap.fromImage(img).scaled(self.orgementaion_view.height(),self.orgementaion_view.width()))
        else:
            self.result[env] = cv2.cvtColor(self.load_result_window.got_img(),cv2.COLOR_BGR2RGB)
            self.orgementaion_view.clear()
            img = QImage(self.result[env], self.orgementaion_view.width(), self.orgementaion_view.height(),self.orgementaion_view.width() * 3, QImage.Format_RGB888)
            self.orgementaion_view.setPixmap(QPixmap.fromImage(img))


if __name__ == '__main__':
    # application 对象
    app = QApplication(sys.argv)

    # QMainWindow对象
    #mainwindow = QMainWindow()

    # 这是qt designer实现的Ui_MainWindow类
    gui = GUI()
    gui.setWindowFlag(Qt.FramelessWindowHint)
    gui.show()
    sys.exit(app.exec_())
