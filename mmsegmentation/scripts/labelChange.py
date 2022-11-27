import cv2
import os
import numpy as np
from PIL import Image

path = "/media/songcc/data/songcc/Retinal/DRIVE/test/2nd_manual"
label_path = '/media/songcc/data/songcc/Retinal/DRIVE/test/2nd_png'
names = os.listdir(path)
label_name = os.listdir(label_path)

# for name in label_name:
#     if not name in names:
#         os.remove(os.path.join(label_path,name))

for name in names:
    img = Image.open(os.path.join(path,name))
    img = np.array(img).astype(np.uint8)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img = img / 255
    # img = img.astype(np.uint8)
    # if np.max(img) == 0:
    #     os.remove(os.path.join(path,name))
    print(np.max(img))
    # ipdb.set_trace()
    cv2.imwrite(os.path.join(label_path,name[:2] + ".png"),img * 255)
