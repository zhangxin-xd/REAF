
# **REAF: Remembering Enhancement and Entropy-based Asymptotic Forgetting for Filter Pruning**
⭐This is a [pytorch](http://pytorch.org/) implementation of REAF.

⭐This paper has been accepted by IEEE TIP.
> ![引用内容](https://github.com/zhangxin-xd/REAF/blob/main/figs/framework.png)
```
@ARTICLE{10181132,
  author={Zhang, Xin and Xie, Weiying and Li, Yunsong and Jiang, Kai and Fang, Leyuan},
  journal={IEEE Transactions on Image Processing}, 
  title={REAF: Remembering Enhancement and Entropy-Based Asymptotic Forgetting for Filter Pruning}, 
  year={2023},
  volume={32},
  number={},
  pages={3912-3923},
  doi={10.1109/TIP.2023.3288986}}
```
This code is based on [ACNet](https://github.com/DingXiaoH/ACNet) and [FPGM](https://github.com/he-y/filter-pruning-geometric-median).
### Prerequisites
- Python 3.8
- Pytorch 1.10.0

### Getting started

- Download [The CIFAR-10 Dataset]( http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

- Download [The CIFAR-100 Dataset]( http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

- Download [The Imagenet Dataset](https://image-net.org/)

### Remembering Enhancement
```
CUDA_VISIBLE_DEVICES=0 python main.py --block_type shadow --pruning_rate =1.0 --weight_path ./checkpoint_pretrain
```

### Entropy-based Asymptotic Forgetting
```
CUDA_VISIBLE_DEVICES=0 python main.py --block_type shadow --pruning_rate =0.6  --weight_path ./checkpoint --pretrain ./checkpoint_pretraint/model_best.pth.tar
```
### Results
![输入图片描述](https://github.com/zhangxin-xd/REAF/blob/main/figs/result1.png)![输入图片描述](https://github.com/zhangxin-xd/REAF/blob/main/figs/result2.png)![输入图片描述](https://github.com/zhangxin-xd/REAF/blob/main/figs/result3.png)
