# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import ipdb

def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    pre_eval=False,
                    format_only=False,
                    format_args={}):

    model.eval()
    dataset = data_loader.dataset
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, **data)
        
        img_tensor = data['img'][0]

        img_metas = data['img_metas'][0].data[0]
        
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        
        assert len(imgs) == len(img_metas)

        for img, img_meta in zip(imgs, img_metas):
            if "bbox" in img_meta.keys():
                bbox = img_meta["bbox"]
                img_res = np.zeros((img_meta["res_shape"][0],img_meta["res_shape"][1])).astype(np.uint8)
                img_res[bbox[0]:bbox[2],bbox[1]:bbox[3]] = result[0]
                result = [img_res]
            res = got_result(result,dataset.PALETTE)    
            return res
def got_result(result,palette=None):
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

