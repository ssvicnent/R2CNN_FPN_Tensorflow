# R2CNN: Rotational Region CNN for Orientation Robust Scene Detection  



## 本代码主要参考中科院大佬的代码，然后在其源码上进行改进,继续向大佬学习！ 
### 大佬的链接如下:  
**[yangyue]**:https://github.com/yangxue0827  
考虑到1080ti显卡显存不够的问题，FPN模块中只用了Resnet101网络中的C2和C4层，并且在做ROIPOOLING的时候只采用7x7的池化尺寸
后续会继续对算法进行改进......  

## 环境配置

* Ubuntu16.04
* Tensorflow-gpu==1.2
* Pyhton2.7
* Opencv-python
* cuda8.0 cudnn5.1
* GTX1080ti  
* ICDAR2015 Dataset
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
将转好的tfrecord放在./data/tfrecord文件夹下  

##ICDAR2015测试效果图如下：
![demo1](./demo_img/1.png)  
