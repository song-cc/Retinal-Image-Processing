import cv2 as cv
import numpy as np

from PyQt5.QtWidgets import QLabel
from PyQt5 import QtGui,QtCore
from PyQt5.QtCore import pyqtSignal

from image_process import reszie
import ipdb
class MyQLabel(QLabel):
    erro_signal = QtCore.pyqtSignal(str)
    def __init__(self,form):
        super(MyQLabel, self).__init__(form)

        self.cur_img = 0
        self.left_flag = False
        self.right_flag = False 

        self.resize_couter = 0
        self.move_x = 0
        self.move_y = 0
        self.center = [self.height() // 2,self.width() // 2]

    # 鼠标左击事件
    def mousePressEvent(self, event):
        y = event.x()
        x = event.y()
        self.starty,self.startx = self.geometry().left(),self.geometry().top()
        # self.endy,self.endx = self.geometry().right(),self.geometry().bottom()
        # print(x,y,"-------------",self.startx,self.starty,self.endx,self.endy)
        # if self.startx < x < self.endx and self.starty < y < self.endy:
        if event.buttons() == QtCore.Qt.LeftButton:
            self.cur_img[x-2:x + 3, y-2:y+3, :] = [255,0,0]

            Image = QtGui.QImage(self.cur_img, self.cur_img.shape[1], self.cur_img.shape[0], 3 * self.cur_img.shape[1],
                                QtGui.QImage.Format_RGB888)

            pixmap = QtGui.QPixmap(Image).scaled(self.w, self.h)
            self.setPixmap(pixmap)

        if event.buttons() == QtCore.Qt.RightButton:
            self.mouse_last_x = y
            self.mouse_last_y = x
            self.right_flag = True

    def wheelEvent(self, a0: QtGui.QWheelEvent) -> None:
        if a0.angleDelta().y() > 0:
            self.resize_couter += 1
        elif a0.angleDelta().y() < 0:
            self.resize_couter -= 1

        if self.resize_couter < 0:
            self.erro_signal.emit("The image is smallest.You can't shrink it.")
            self.resize_couter = 0
            return
        w_crop = int(self.h * 0.9 ** self.resize_couter)
        h_crop = int(self.w * 0.9 ** self.resize_couter)

        show_img,_ = reszie(self.cur_img, [self.center[0], self.center[1]], [w_crop, h_crop])
        show_img = cv.resize(show_img, (self.w, self.h))
        Image = QtGui.QImage(show_img.data, show_img.shape[1], show_img.shape[0],
                             3 * show_img.shape[1], QtGui.QImage.Format_RGB888)

        pixmap = QtGui.QPixmap(Image).scaled(self.w, self.h)
        self.setPixmap(pixmap)


    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        y = ev.x()
        x = ev.y()
        if self.right_flag:
            cur_y = ev.y()
            cur_x = ev.x()

            self.move_x = int((cur_x - self.mouse_last_x) * 0.9 ** self.resize_couter)
            self.move_y = int((cur_y - self.mouse_last_y) * 0.9 ** self.resize_couter)

            w_crop = int(self.h * 0.9 ** self.resize_couter)
            h_crop = int(self.w * 0.9 ** self.resize_couter)

            self.center[0] -= self.move_y
            self.center[1] -= self.move_x

            show_img,self.center = reszie(self.cur_img, [self.center[0], self.center[1]], [w_crop, h_crop])
            show_img = cv.resize(show_img, (self.w, self.h))
            Image = QtGui.QImage(show_img.data, show_img.shape[1], show_img.shape[0],
                                3 * show_img.shape[1], QtGui.QImage.Format_RGB888)

            pixmap = QtGui.QPixmap(Image).scaled(self.w, self.h)
            self.setPixmap(pixmap)

class segmentationView(QLabel):
    axies_signal = pyqtSignal(list)
    def __init__(self,form):
        super(segmentationView, self).__init__(form)
        
        self.call = False
        self.form = form
        self.cur_img = 0
        self.x = -1
        self.y = -1
        self.center = [self.height() // 2,self.width() // 2]
        self.vessel_segment = None
        self.axies_signal.connect(self.form.after_diameter_estimate_clicked)


    # 鼠标左击事件,重写函数
    def mousePressEvent(self, event):
        if self.call:
            self.call = False
            y = event.x()
            x = event.y()
            self.axies_signal.emit([x,y])

    
    def get_location(self):
        return [self.x,self.y]
    