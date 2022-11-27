# Retinal-Image-Processing
This project is a color fundus image analysis system based on deep learning.It can achieve optic disc and cup segmentation, blood vessel segmentation, and arteri/venous segmentation. And it can measure optic cup and disc diameter; cup-to-disc ratio; blood vessel diameter; CRAE and CRVE; blood vessel curvature; blood vessel bifurcation angle; blood vessel fractal dimension and so on.It should be noted that, except for the segmentation method, the measurement of these parameters is based on the reproduction of existing papers, and there is no strict comparative experiment to prove its accuracy.

## How to Start
### 1. Environment Instructions
The development of the whole system is developed on the Ubuntu20.4 platform, and the GPU used is 1080Ti. I tested it on an ubuntu system with a 3060GPU, and there is no problem. Although I haven't tested it on Windows and Mac systems, I think there will be no serious bugs.In order to run the program smoothly, I have configured a virtual environment in anaconda, which can be configured through the following commands.

```shell
conda env create -f enveriment.yaml 
```

### Download Project and Model
You can download the project and [pretrained model](https://share.weiyun.com/lj6oX7hQ). And then place them in the corresponding directory according to different models. The directory in my development process looks like [this](./Retinal_GUI.txt).


### Run
you can run this project with this command

```
conda activate GUI
python Retinal_GUI_main.py
```

## References
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
[SA-UNet](https://github.com/clguo/SA-UNet)
[CE-Net](https://github.com/Guzaiwang/CE-Net)
