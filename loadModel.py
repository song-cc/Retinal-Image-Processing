import mmcv
import torch
import numpy as np
import torch
import cv2
from torch.autograd import Variable as V

from Stage1.UNet import UNet
from ROI_Detection.ResNet import ResNet101
from Stage2.cenet import CE_Net_
from SAUNet.SAUNet import SA_UNet
from mmsegmentation.mmseg.apis import inference_segmentor, init_segmentor
# import mmcv
# from mmsegmentation.mmseg.utils import setup_multi_processes
# from mmcv.runner import get_dist_info,init_dist,load_checkpoint,wrap_fp16_model

class Stage1_SegMentor(torch.nn.Module):
    def __init__(self,device,model_path):
        super(Stage1_SegMentor, self).__init__()
        self.device = device
        self.model = UNet([3, 64, 128, 256, 512], [512, 1024, 512, 256, 128])
        model_weight = torch.load(model_path)
        self.model.load_state_dict(model_weight)
        self.model.eval()
        self.model = self.model.to(self.device)

    def __call__(self, img_tensor,shape,*args, **kwargs):
        img_tensor = img_tensor.to(self.device)
        img_tensor = img_tensor.permute(0, 3, 1, 2)
        img_tensor = img_tensor.float()
        _,_,_,_,y_hat = self.model(img_tensor)
        y_0 = y_hat.data.cpu().numpy()
        res = self.original_img(y_0,shape)

        return res

    def original_img(self,onehot, x_shape, save=False, save_path=None):
        _, _, length, width = np.shape(onehot)

        curr = onehot[0, :, :, :]
        curr.resize((length, width))
        Thread = np.max(curr)
        img = np.zeros([length, width])
        img[curr > 0.2 * Thread] = 255
        img = cv2.resize(img, (x_shape[0], x_shape[1]))
        img[img > 10] = 255

        if save == True:
            cv2.imwrite(save_path, img)

        return img

class Roi_Detector(torch.nn.Module):
    def __init__(self,device,model_path):
        super(Roi_Detector, self).__init__()
        self.img_size = 224
        self.device = device
        self.model = ResNet101()
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.eval()
    def  __call__(self, imgs,*args, **kwargs):
        n, _, _, _ = np.shape(imgs)
        x = np.zeros((n,self.img_size,self.img_size,3))

        for i in range(n):
            x[i,:,:] = cv2.resize(imgs[i],(self.img_size,self.img_size))

        x = torch.from_numpy(x)
        x = x.float()
        x = x.permute(0, 3, 1, 2)
        x = x.to(self.device)
        y_hat = self.model(x)
        y_0 = y_hat.data.cpu().numpy()

        for i in range(n):
            if y_0[i] >= max(y_0):
                return i

class Stage2_SegMentor(torch.nn.Module):
    def __init__(self,device,model_path):
        super(Stage2_SegMentor, self).__init__()
        self.deivce = device
        self.model = CE_Net_()
        self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.deivce)
        self.model.eval()

    def __call__(self, img, *args, **kwargs):
        w,h,_ = np.shape(img)
        img = cv2.resize(img,(256,256))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0
        img5 = V(torch.Tensor(img5).cuda())

        mask,_ = self.model.forward(img5)
        mask = mask.squeeze().cpu().data.numpy()  # .squeeze(1)
        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]
        mask3 = cv2.resize(mask3,(w,h))
        thread = np.max(mask3) * 0.2
        _,mask3 = cv2.threshold(mask3,thread,255,cv2.THRESH_BINARY)

        return mask3

class sa_UNet():
    def __init__(self,model_path) -> None:
        self.model = SA_UNet(input_size=(592,592,3),start_neurons=16,lr=1e-3,keep_prob=0.82,block_size=7)
        self.model.load_weights(model_path)

class mmsegmentor():
    def __init__(self,config,checkpoint) -> None:
        device = "cuda:0"
        self.model = init_segmentor(config, checkpoint, device=device)

    def __call__(self,img,palette):
        res = inference_segmentor(self.model, img)
        res = self.got_result(res,palette)
        return res
        
    def got_result(self,result,palette=None):
        seg = result[0]
        palette = np.array(palette)
        if len(palette[0]) == 3:
            color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
            for label, color in enumerate(palette):
                color_seg[seg == label, :] = color
        if len(palette[0]) == 1:
            color_seg = np.zeros((seg.shape[0], seg.shape[1]), dtype=np.uint8)
            for label, color in enumerate(palette):
                color_seg[seg == label] = color
        return color_seg
