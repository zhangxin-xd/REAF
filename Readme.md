
# **REAF: Remembering Enhancement and Entropy-based Asymptotic Forgetting for Filter Pruning**
This is a [pytorch](http://pytorch.org/) implementation of REAF.
> ![引用内容](\figs\framework.png)

### Prerequisites
- Python 3.8
- GPU Memory >= 11G
- Pytorch 1.10.0

### Getting started

- Download [The CIFAR-10 Dataset]( http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

- Download [The CIFAR-100 Dataset]( http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

- Download [The Imagenet Dataset](https://image-net.org/)

### Remembering Enhancement
```
CUDA_VISIBLE_DEVICES=0 python main.py --block_type shadow --af False --pruning_rate =1.0
```

### Entropy-based Asymptotic Forgetting
```
CUDA_VISIBLE_DEVICES=0 python main.py --block_type shadow --af True --pruning_rate =0.6
```
### Results
![输入图片描述](Readme_md_files/79cefdc0-1950-11ed-a285-d1ceb57c1d9c.jpeg?v=1&type=image)![输入图片描述](Readme_md_files/80c92f60-1950-11ed-a285-d1ceb57c1d9c.jpeg?v=1&type=image)![输入图片描述](Readme_md_files/965e8b90-1950-11ed-a285-d1ceb57c1d9c.jpeg?v=1&type=image)
