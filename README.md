# Personal Protective Equipment Detection

## Table of contents
1. [Introduction](#introduction)
2. [Methodology overview](#methodology)
	- [Dummy neck & Modified simple-FPN](#neck)
	- [Model architecture](#model)
3. [Settings](#setting)
    - [Pre-train phase](#pretrain)
    - [Fine-tunning phase](#finetune)
4. [Experiment results](#result)
    - [Val2017](#val1)
    - [CHVGVal](#val2)
5. [Analysis (but not really)](#analysis)
    - [Learning rate](#lr)
    - [Training loss](#loss)
    - [Decoder output](#output)
    - [Haze, Rain, Low-light effect on CHVGVal](#effect)
6. [Web application](#web)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Demo](#demo)

## <a name="introduction"></a> Introduction
About this repository:
- It was built based on the official implementation of the paper [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://github.com/fundamentalvision/Deformable-DETR). 
- This is the summary of our final project at FPT university. For more details about the project, please take a look at our dissertation [here!](https://drive.google.com/drive/folders/1kZmqb7pM0cy5Ww1Swt6KqmZImHyLHo6y?usp=sharing)
- In this project, we try different ways to assemble models based on the two currently state-of-the-art approaches in Fewshot Object Detection, which are [imTED](https://arxiv.org/abs/2205.09613) and [DETReg](https://arxiv.org/abs/2106.04550). With the hope that it can be applied to PPE detection tasks in low-data regimes.
- If you're looking for a model that can actually work, this is definitely not the place. Since there are lots of limitations in computation resources and time, our models haven't converged yet; furthermore, lots of training settings and architecture are modified so that each epoch can be finished within a Kaggle session time limit.

## <a name="methodology"></a> Methodology overview
### <a name="neck"></a> Dummy neck & Modified simple-FPN
The first neck we represented is called "dummy" neck that use the output of the last two layers from ViT to generate four feature maps. The first two are obtained by using a single linear mapping that projects the dimension of feature from 768 to 256. The remains are produced by feeding the outputs through a vanilla FPN with lateral and top-down connections.

![Fig. 12](/static/Fig12.png)

The other is a modified version of simple-FPN, which was proposed in the paper [Exploring Plain Vision Transformer Backbones for Object Detection](https://arxiv.org/abs/2203.16527). The modification is made due to the limitation in our computing resources: the deconvolution of stride 4 is replaced by a convolution of stride 4.

![Fig. 13](/static/Fig13.png)

### <a name="model"></a> Model architecture
Using the ideas above, we then represent three combinations called model_0, model_1, model_2. Their architecture is as follows:

![Fig. 14](/static/Fig14.png)

A comparison between these models is shown below:

|    Model    |      Neck      |  #Params  |   FPS   |
|:-------------:|:----------------:|:-----------:|:---------:|
| Model_0 | Dummy neck |  96.3M  | 9.4 |
| Model_1 | Dummy neck |  100M   | 7.9 |
| Model_2 | Modified sFPN | 103.4M   | 7.4 |

## <a name="setting"></a> Settings
### <a name="pretrain"></a> Pre-train phase
Again, due to the limitation in computation resources, we train all of the models using [miniCOCO](https://github.com/giddyyupp/coco-minitrain), which is a curated mini training set for [COCO](https://cocodataset.org/#home). Other details about data augmentation and optimization are presented in our [thesis](https://drive.google.com/drive/folders/1kZmqb7pM0cy5Ww1Swt6KqmZImHyLHo6y?usp=sharing)!

## <a name="finetune"></a> Fine-tunning phase
The data that we used was created by manually choosing images from [CHVGTrain](https://universe.roboflow.com/scalersai/chvg-conversion).

During fine-tunning phase, only the detector head is finetuned while the remaining components are frozen.

## <a name="result"></a> Experiment results
### <a name="val1"></a> Val2017
| Method        | Backbone         | AP    | AP\_50 | AP\_75 | AP\_S | AP\_M | AP\_L |
|---------------|------------------|-------|--------|--------|-------|-------|-------|
| Faster R\-CNN | ResNet\-50 w FPN | 27\.7 | 48\.8  | 28\.4  | 14\.7 | 29\.8 | 36\.4 |
| Mask R\-CNN   | ResNet\-50 w FPN | 28\.5 | 49\.5  | 29\.4  | 14\.7 | 30\.7 | 37\.6 |
| RetinaNet     | ResNet\-50 w FPN | 25\.7 | 43\.1  | 26\.8  | 12\.1 | 28\.6 | 34\.2 |
| CornerNet     | Hourglass\-104   | 28\.4 | 41\.8  | 29\.5  | 11\.3 | 29\.6 | 39\.2 |
| ExtremeNet    | Hourglass\-104   | 27\.3 | 39\.4  | 28\.9  | 12\.5 | 29\.6 | 38\.0 |
| Model_0       | ViT, MAE         | 1\.7  | 4\.0   | 1\.1   | 0\.4  | 1\.3  | 2\.8  |
| Model_1       | ViT, MAE         | 7\.9  | 14\.2  | 7\.7   | 1\.4  | 6\.5  | 14\.8 |
| Model_2       | ViT, MAE         | 6\.3  | 11\.5  | 6\.1   | 0\.7  | 5\.2  | 11\.6 |

### <a name="val2"></a> CHVGVal
| Method        | Backbone         | AP    | AP\_50 | AP\_75 | AP\_S | AP\_M | AP\_L |
|---------------|------------------|-------|--------|--------|-------|-------|-------|
| DDETR         | ResNet\-50       | 38\.7 | 67\.6  | 38\.7  | 12\.8 | 39\.3 | 48\.4 |
| DETReg        | ResNet\-50       | 39\.5 | 68\.0  | 39\.9  | 12\.2 | 38\.9 | 50\.5 |
| Model_0       | ViT, MAE         | 0\.3  | 1\.0   | 0\.0   | 0\.0  | 0\.0  | 0\.4  |
| Model_1       | ViT, MAE         | 1\.4  | 3\.0   | 1\.2   | 0\.0  | 0\.9  | 1\.6  |
| Model_2       | ViT, MAE         | 0\.4  | 1\.1   | 0\.2   | 0\.0  | 0\.0  | 0\.5  |

## <a name="analysis"></a> Analysis (but not really)
### <a name="lr"></a> Learning rate
After trying different settings about learning rate, we hypothesize that, although ViT has been pretrained under a self-supervised task, it is not absolutely align to object detection. Thus, a larger learning rate is 
obvious. In contrast, DDETR's transformer has been pretrained under self-supervision task 
that’s closer to object detection, so higher learning rate is not needed.
 
### <a name="loss"></a> Training loss
As shown in figure below, all three of our models are continuing to 
converge. Model_1 and model_2 give quite similar results, while model_0 converges 
slower and takes more time. However, due to time constraints, we have not been able to 
train these models to the point of convergence. Therefore, it is unable to assess whether the 
miniCOCO dataset is enough to pretrain transformer architecture models, which are well-known as "hungry data" models.

![Fig. 15](/static/Fig15.png)

### <a name="output"></a> Decoder output
Similar to DETRs, each query learns to specialize in certain areas and box sizes.

![Fig. 16](/static/Fig16.png)

We visualize the attention maps of the last 
encoder layer of trained models. Each attention map corresponds to a feature map 
generated from the model's neck.

![Fig. 17](/static/Fig17.png)

### <a name="effect"></a> Haze, Rain, Low-light effect on CHVGVal
In this part, we generate image with weather effect follow the [article](https://peerj.com/articles/cs-999/), then reasoning about the results. As this section is too long for a summary, please see [here](https://drive.google.com/drive/folders/1kZmqb7pM0cy5Ww1Swt6KqmZImHyLHo6y?usp=sharing).


## <a name="web"></a> Web application
### <a name="installation"></a> Installation
- First, please install the environment follow [this](https://github.com/amirbar/DETReg).
- Then install other requirements as below:
```bash
    pip install timm==0.3.2
    pip install Flask
```

### <a name="usage"></a> Usage
In order to run app.py, please organize the reposity as follow:
```
code_root/
├── mDETD/
└── pre-trained checkpoints/
    ├── mDETD_0_.pth
    ├── mDETD_1_.pth
    ├── mDETD_2_.pth
    ├── DDETR_.pth
    └── DETReg_.pth
```


### <a name="demo"></a> Demo
The web application give you some options, you can choose the model (either model_0, model_1, model_2, DDETR, DETReg), and the demo type (either from image or live cam).

![Screen1](/static/web1.png)

Below is the result we got after submit an image.

![Screen2](/static/web2.png)

And from the camera.

![Screen3](/static/web3.png)
