# R2CNN_FPN_Tensorflow


![demo](./demo_img/1.png)
# R2CNN: Rotational Region CNN for Orientation Robust Scene Detection
## 本代码主要参考中科院大佬的代码，然后在其源码上进行改进和修饰,继续向大佬学习！
### 大佬的链接如下：[yangyue]:https://github.com/yangxue0827
考虑到1080ti显卡显存不够的问题，FPN模块中只用了Resnet101网络中的C2和C4层，并且在做ROIPOOLING的时候只采用7x7的池化尺寸
后续会继续改进......


##环境配置

* Ubuntu16.04
* Tensorflow-gpu==1.2
* Pyhton2.7
* Opencv-python
* cuda8.0 cudnn5.1
* GTX1080ti
