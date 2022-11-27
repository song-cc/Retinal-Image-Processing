# Retinal-Image-Processing
This project is a color fundus image analysis system based on deep learning.It can achieve optic disc and cup segmentation, blood vessel segmentation, and arteri/venous segmentation. And it can measure optic cup and disc diameter; cup-to-disc ratio; blood vessel diameter; CRAE and CRVE; blood vessel curvature; blood vessel bifurcation angle; blood vessel fractal dimension and so on.It should be noted that, except for the segmentation method, the measurement of these parameters is based on the reproduction of existing papers, and there is no strict comparative experiment to prove its accuracy.

## How to Start
### 1. Environment Instructions
The development of the whole system is developed on the Ubuntu20.4 platform, and the GPU used is 1080Ti. I tested it on an ubuntu system with a 3060GPU, and there is no problem. Although I haven't tested it on Windows and Mac systems, I think there will be no serious bugs.In order to run the program smoothly, I have configured a virtual environment in anaconda, which can be configured through the following commands.

'''
conda env create -f enveriment.yaml 
'''

### Download Project and Model
You can download the project and [pretrained model](https://share.weiyun.com/lj6oX7hQ). And then place them in the corresponding directory according to different models. The directory in my development process looks like this.

main
├── 1-最大化.png
├── backgroud.qrc
├── backgroud_rc.py
├── image_process.py
├── inflections.py
├── loadModel.py
├── logo.png
├── measurement.py
├── mmsegmentation
│   ├── mmseg
│   │   ├── apis
│   │   │   ├── inference.py
│   │   │   ├── __init__.py
│   │   │   ├── test.py
│   │   │   └── train.py
│   │   ├── core
│   │   │   ├── builder.py
│   │   │   ├── evaluation
│   │   │   │   ├── class_names.py
│   │   │   │   ├── eval_hooks.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── metrics.py
│   │   │   ├── __init__.py
│   │   │   ├── layer_decay_optimizer_constructor.py
│   │   │   ├── optimizers
│   │   │   │   ├── __init__.py
│   │   │   │   ├── layer_decay_optimizer_constructor.py
│   │   │   ├── seg
│   │   │   │   ├── builder.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── sampler
│   │   │   │       ├── base_pixel_sampler.py
│   │   │   │       ├── __init__.py
│   │   │   │       ├── ohem_pixel_sampler.py
│   │   │   └── utils
│   │   │       ├── dist_util.py
│   │   │       ├── __init__.py
│   │   │       ├── layer_decay_optimizer_constructor.py
│   │   │       ├── misc.py
│   │   ├── datasets
│   │   │   ├── builder.py
│   │   │   ├── cityscapes.py
│   │   │   ├── custom.py
│   │   │   ├── dataset_wrappers.py
│   │   │   ├── drive.py
│   │   │   ├── drive_Segformer.py
│   │   │   ├── __init__.py
│   │   │   ├── pipelines
│   │   │   │   ├── bondary_map.py
│   │   │   │   ├── compose.py
│   │   │   │   ├── formating.py
│   │   │   │   ├── formatting.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── loading.py
│   │   │   │   ├── test_time_aug.py
│   │   │   │   └── transforms.py
│   │   │   └── samplers
│   │   │       ├── distributed_sampler.py
│   │   │       ├── __init__.py
│   │   ├── __init__.py
│   │   ├── models
│   │   │   ├── backbones
│   │   │   │   ├── __init__.py
│   │   │   │   ├── mit.py
│   │   │   │   └── vit.py
│   │   │   ├── builder.py
│   │   │   ├── decode_heads
│   │   │   │   ├── decode_head.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── segformer_head.py
│   │   │   ├── __init__.py
│   │   │   ├── losses
│   │   │   │   ├── accuracy.py
│   │   │   │   ├── ce_net_loss
│   │   │   │   ├── cross_entropy_loss.py
│   │   │   │   ├── dice_loss.py
│   │   │   │   ├── focal_loss.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── lovasz_loss.py
│   │   │   │   └── utils.py
│   │   │   ├── necks
│   │   │   │   ├── featurepyramid.py
│   │   │   │   ├── fpn.py
│   │   │   │   ├── ic_neck.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── jpu.py
│   │   │   │   ├── mla_neck.py
│   │   │   │   ├── multilevel_neck.py
│   │   │   ├── segmentors
│   │   │   │   ├── base.py
│   │   │   │   ├── cascade_encoder_decoder.py
│   │   │   │   ├── encoder_decoder.py
│   │   │   │   ├── __init__.py
│   │   │   └── utils
│   │   │       ├── embed.py
│   │   │       ├── __init__.py
│   │   │       ├── inverted_residual.py
│   │   │       ├── make_divisible.py
│   │   │       ├── res_layer.py
│   │   │       ├── se_layer.py
│   │   │       ├── self_attention_block.py
│   │   │       ├── shape_convert.py
│   │   │       └── up_conv_block.py
│   │   ├── ops
│   │   │   ├── encoding.py
│   │   │   ├── __init__.py
│   │   │   └── wrappers.py
│   │   ├── utils
│   │   │   ├── collect_env.py
│   │   │   ├── __init__.py
│   │   │   ├── logger.py
│   │   │   ├── misc.py
│   │   │   └── set_env.py
│   │   └── version.py
│   └── work_dirs
│       ├── segformer_mit-b4_512x512_160k_AVdrive_Original
│       │   ├── iter_80000.pth
│       │   └── segformer_mit-b4_512x512_160k_AVdrive.py
│       ├── segformer_mit-b4_512x512_160k_drive
│       │   ├── iter_80000.pth
│       │   └── segformer_mit-b4_512x512_160k_drive.py
│       └── segformer_mit-b4_512x512_160k_driveMulti
│           ├── iter_80000.pth
│           └── segformer_mit-b4_512x512_160k_driveMulti.py
├── myQTClass.py
├── Retail_GUI.py
├── Retinal_GUI_main.py
├── Retinal_GUI.txt
├── ROI_Detection
│   ├── ResNet.py
│   └── ROI_Detector.param
├── SAUNet
│   ├── Dropblock.py
│   ├── Model
│   │   └── SA_UNet.h5
│   ├── SAUNet.py
│   ├── Spatial_Attention.py
│   └── test.py
├── Stage1
│   ├── DropBlock.py
│   ├── SA_block.py
│   ├── Stage1.params
│   └── UNet.py
├── Stage2
│   ├── cenet.py
│   ├── CE-Net.th
│   └── W-Net-cup.pth
├── T0013.png
├── 关闭.png
└── 最小化.png


### Run
you can run this project with this command

'''
conda activate GUI
python Retinal_GUI_main.py
'''

## References
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
[SA-UNet](https://github.com/clguo/SA-UNet)
[CE-Net](https://github.com/Guzaiwang/CE-Net)
