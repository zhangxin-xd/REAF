from models.mobilenetv1 import *
from models.stagewise_resnet import *
from models.vgg import *
from models.lenet5 import create_lenet5bn
from models.wrn import create_wrnc16plain
from models.resnet import create_ResNet18, create_ResNet34
from models.cfqk import create_CFQKBNC

IMAGENET_STANDARD_MODEL_MAP = {
    'sres50': create_SResNet50,
    'sres101': create_SResNet101,
    'smi1': create_MobileV1Imagenet,
    'sres18': create_ResNet18,
    'sres34': create_ResNet34
}

CIFAR10_MODEL_MAP = {
    'src56':create_SRC56,
    'src110':create_SRC110,
    'vc':create_vc,
    'wrnc16plain':create_wrnc16plain,
    'cfqkbnc':create_CFQKBNC
}
CIFAR100_MODEL_MAP = {
    'src56':create_SRC56_100,
    'src110':create_SRC110_100,
}
MNIST_MODEL_MAP = {
    'lenet5bn': create_lenet5bn,
}

DATASET_TO_MODEL_MAP = {
    'imagenet_standard': IMAGENET_STANDARD_MODEL_MAP,
    'cifar10': CIFAR10_MODEL_MAP,
    'cifar100': CIFAR100_MODEL_MAP,

}


#   return the model creation function
def get_model_fn(dataset_name, model_name):
    # print(DATASET_TO_MODEL_MAP[dataset_name.replace('_blank', '_standard')].keys())
    return DATASET_TO_MODEL_MAP[dataset_name][model_name]

def get_dataset_name_by_model_name(model_name):
    for dataset_name, model_map in DATASET_TO_MODEL_MAP.items():
        if model_name in model_map:
            return dataset_name
    return None