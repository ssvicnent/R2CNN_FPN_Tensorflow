# R2CNN: Rotational Region CNN for Orientation Robust Scene Detection  

本代码主要参考中科院大佬的代码，然后在其源码上进行改进,继续向大佬学习！   
大佬的链接为[yangyue0826](https://github.com/yangxue0827) 。  
考虑到1080ti显卡显存不够的问题，FPN模块中只用了Resnet101网络中的C2和C4层，并对FPN进行改进，除此之外在做roipooling的时候只采用7x7的池化尺寸，后续会继续对算法进行改进...  
* C2和C4层中的RPN的配置参数是不同的，C4层由于语义信息比较丰富，但是文本位置比较粗糙，因此设置Anchors的比例数相对来说多一些;
* C2层语义信息没有高层的C4层的丰富，但是位置比较精确，能够将小的文本目标检测出来，因此C2层的设置Anchors的比例数比C4层的少；
* C2和C4层的RPN参数具体参照`<./libs/configs/cfgs.py>`。

## 环境配置

* Ubuntu16.04
* Tensorflow-gpu==1.2
* Pyhton2.7
* Opencv-python
* cuda8.0 cudnn5.1
* GTX1080ti  
* [ICDAR2015 Dataset](http://rrc.cvc.uab.es/)  
## 将数据集转化为TFrecord格式  
数据要求的是VOC的格式，VOC的格式如下：  
```
├── VOCdevkit
│   ├── VOCdevkit_train
│       ├── Annotation
│       ├── JPEGImages
│    ├── VOCdevkit_test
│       ├── Annotation
│       ├── JPEGImages
```  
代码如下:
```
cd ./data/io/
python convert_data_to_tfrecord.py --VOC_dir='***/VOCdevkit/VOCdevkit_train/' --save_name='train' --img_format='.jpg' --dataset='icdar2015'  
```
将转好的tfrecord放在`<./data/tfrecord>`文件夹下  
## 编译(Compile)
```
cd ./libs/box_utils/
python setup.py build_ext --inplace
```  

## Train and Test
1.下载预训练的ResNet101的checkpoint为[resnet_v1_101_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz)，然后将其解压到文件夹`</data/pretrained_weights>`下  
2.训练的时候：

```
cd ./tools/
python train.py
```  
3.测试的时候：
```
cd ./tools/
python inference.py --data_dir=inference_image --gpu='0'
```  

## Summary
训练的时候可通过Tensorboard来Loss function的动态，以及训练过程的实时效果图
```
cd ./output/summary/
tensorboard --logdir=./
```
## ICDAR2015测试效果图如下：
![demo1](./demo_img/1.png)  
