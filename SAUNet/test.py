from SA_UNet import *
import os
import cv2
import random
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np

desired_size = 592

def load_image(img_path):
    im = cv2.imread(img_path)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

    old_size = im.shape[:2]  # old_size is in (height, width) format
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    img_use = cv2.resize(new_im, (desired_size, desired_size))
    return img_use,[top,bottom,left,right],old_size

def load_data_infer(image_path):
    img,pi_index,shapes = load_image(image_path)

    return img,shapes,pi_index

model=SA_UNet(input_size=(desired_size,desired_size,3),start_neurons=16,lr=1e-3,keep_prob=0.82,block_size=7)
weight="Model/DRIVE/SA_UNet.h5"
model.load_weights(weight)

test_images_loc = '/media/songcc/data/songcc/Retinal/DRIVE/training/image_png'
test_res_dir = 'SA_UNet-train'

if not os.path.isdir(test_res_dir):
    os.mkdir(test_res_dir)
test_files = os.listdir(test_images_loc)
test_data = []

for i in test_files:

    new_im,shapes,pixel_index = load_data_infer(os.path.join(test_images_loc,i))

    print('*******************',shapes)
    start_x = int(pixel_index[0])
    end_x = int(pixel_index[1])
    start_y = int(pixel_index[2])
    end_y = int(pixel_index[3])


    new_im = new_im[np.newaxis,...]
    x_test = new_im.astype('float32') / 255.
    y_hat = model.predict(x_test)
    y = y_hat.reshape((desired_size,desired_size))
    res = np.zeros((desired_size,desired_size))
    res[y >= 0.5] = 1
    res[y < 0.5] = 0
    res = res * 255
    out = res[start_x:-end_x,start_y:-end_y]
    # res = cv2.resize(res, (end_y - start_y, end_x - start_x))
    # out[start_x:-end_x,start_y:-end_y] = res
    cv2.imwrite(os.path.join(test_res_dir,i[:-4] + '.png'),out)
    # print(np.shape(out))
'''
    test_data.append(cv2.resize(new_im, (desired_size, desired_size)))

test_data = np.array(test_data)


x_test= test_data.astype('float32') / 255.
x_test = np.reshape(x_test, (
len(x_test), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format

TensorBoard(log_dir='./autoencoder', histogram_freq=0,
            write_graph=True, write_images=True)

predict = model.predict(x_test)

n,_,_ = np.shape(predict)

print('*******hello')
print(np.shape(predict))
'''
