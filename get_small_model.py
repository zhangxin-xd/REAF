import argparse
import os,sys
import shutil
import pdb, time
from collections import OrderedDict
import torchvision.datasets as dset
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from test import general_test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import convert_secs2time, time_string, time_file_str
# from models import print_log
import models_deploy
import random
import numpy as np
import copy
from model_cfg import get_model_fn
from creaters.shadow_creater import ShadowCreater
import thop
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_names = sorted(name for name in models_deploy.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models_deploy.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_path', metavar='DIR', default = '',help='path to dataset')
parser.add_argument('--save_dir', type=str, default='', help='Folder to save checkpoints and log.')
parser.add_argument('--dataset', type=str, default='cifar10', help='Choose from cifar10 and cifar 100')
parser.add_argument('--arch', type=str, default='src56', help='Choose from ')
# parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet56', choices=model_names,
#                     help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--print-freq', '-p', default=1, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--layer_begin', type=int, default=3, help='compress layer of model')
parser.add_argument('--layer_end', type=int, default=3, help='compress layer of model')
parser.add_argument('--layer_inter', type=int, default=1, help='compress layer of model')
parser.add_argument('--epoch_prune', type=int, default=1, help='compress layer of model')
parser.add_argument('--skip_downsample', type=int, default=1, help='compress layer of model')
parser.add_argument('--get_small',default=True, dest='get_small', action='store_true', help='whether a big or small model')

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

args.prefix = time_file_str()


def main():
    best_prec1 = 0

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    log = open(os.path.join(args.save_dir, 'gpu-time.{}.{}.log'.format(args.arch, args.prefix)), 'w')

    # create model
    print_log("=> creating model '{}'".format(args.arch), log)
    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    if args.dataset == 'cifar10':
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 100

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    model = models_deploy.__dict__[args.arch](num_classes)
    # model = torch.nn.DataParallel(model)
    print_log("=> Model : {}".format(model), log)
    print_log("=> parameter : {}".format(args), log)
    print_log("Layer Begin: {}".format(args.layer_begin), log)
    print_log("Layer End: {}".format(args.layer_end), log)
    print_log("Layer Inter: {}".format(args.layer_inter), log)
    print_log("Epoch prune: {}".format(args.epoch_prune), log)
    print_log("Skip downsample : {}".format(args.skip_downsample), log)
    target_weights = os.path.join(args.save_dir, 'best.pth')
    from fusion import convert_weights
    criterion = nn.CrossEntropyLoss().cuda()
    convert_weights(target_weights, target_weights.replace('.pth', '_deploy.pth'), eps=1e-5)
    deploy_creater = ShadowCreater(deploy=True)
    general_test(args=args, weights=target_weights.replace('.pth', '_deploy.pth'),
                 creater = deploy_creater, test_loader=test_loader, criterion = criterion)
    args.weight = os.path.join(args.save_dir, 'best_deploy.pth')
    # optionally resume from a checkpoint
    if args.weight:
        if os.path.isfile(args.weight):
            print_log("=> loading checkpoint '{}'".format(args.weight), log)
            checkpoint = torch.load(args.weight)
            new_state_dict = {}
            for key in checkpoint:
                if 'stage1.conv1.fused_conv' in key:
                    newkey = key.replace('stage1.conv1.fused_conv','conv_1_3x3.conv')
                    new_state_dict[newkey] = checkpoint[key]
                elif 'stage' in key:
                    newkey = key.replace('stage','stage_')
                    if 'block' in key:
                        newkey = newkey.replace('block', '')
                        if 'conv2.fused_conv' in key:
                            newkey = newkey.replace('conv2.fused_conv','conv_b')
                            new_state_dict[newkey] = checkpoint[key]
                        if 'conv1.acb.fused_conv' in key:
                            newkey = newkey.replace('conv1.acb.fused_conv','conv_a')
                            new_state_dict[newkey] = checkpoint[key]
                else:
                    newkey = key.replace('linear','classifier')
                    new_state_dict[newkey] = checkpoint[key]
            model.load_state_dict(new_state_dict)
            print_log("=> loaded checkpoint '{}' ".format(args.weight), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.weight), log)


    criterion = nn.CrossEntropyLoss().cuda()


    if args.get_small:
        big_path = os.path.join(args.save_dir, "big_model.pt")
        torch.save(model, big_path)
   
        small_model = get_small_model(model)
        
        # small_model = torch.load('small_model.pt')
        small_path = os.path.join(args.save_dir, "small_model.pt")
        torch.save(small_model, small_path)

        if args.use_cuda:
            model = model.cuda()
            small_model = small_model.cuda()
        print('evaluate: big')
        top1_big = validate(test_loader, model, criterion, log)
        

        print('evaluate: small')
        top1_small = validate(test_loader, small_model, criterion, log)
        print('big model accu', top1_big)
        print('small model accu', top1_small)

        from thop import profile
        input_image = torch.randn(1, 3, 256, 256).cuda()
        flops, params = profile(small_model, inputs=(input_image,))

        print('Params: %.2f'%(params))
        print('Flops: %.2f'%(flops))


def validate(test_loader, model, criterion, log):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            target = target.cuda()
            # if i>0:
            #     break
            if args.use_cuda:
                input, target = input.cuda(), target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print_log('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(test_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5), log)

    print_log(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                           error1=100 - top1.avg), log)

    return top1.avg


def save_checkpoint(state, is_best, filename, bestname):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def remove_module_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def check_channel(tensor):
    size_0 = tensor.size()[0]
    size_1 = tensor.size()[1] * tensor.size()[2] * tensor.size()[3]
    tensor_resize = tensor.view(size_0, -1)
    # indicator: if the channel contain all zeros
    channel_if_zero = np.zeros(size_0)
    for x in range(0, size_0, 1):
        channel_if_zero[x] = np.count_nonzero(tensor_resize[x].cpu().numpy()) != 0
    # indices = (torch.LongTensor(channel_if_zero) != 0 ).nonzero().view(-1)

    indices_nonzero = torch.LongTensor((channel_if_zero != 0).nonzero()[0])
    # indices_nonzero = torch.LongTensor((channel_if_zero != 0).nonzero()[0])

    zeros = (channel_if_zero == 0).nonzero()[0]
    indices_zero = torch.LongTensor(zeros) if zeros != [] else []

    return indices_zero, indices_nonzero


def extract_para(big_model):
    '''
    :param model:
    :param batch_size:
    :return: num_for_construc: number of remaining filter,
             [conv1,stage1,stage1_expend,stage2,stage2_expend,stage3,stage3_expend,stage4,stage4_expend]

             kept_filter_per_layer: number of remaining filters for every layer
             kept_index_per_layer: filter index of remaining channel
             model: small model
    '''
    # resnet50 
    item = list(big_model.state_dict().items())
    print("length of state dict is", len(item))
    #print(item)
    try:
        assert len(item) in [112, ]  # 112 resnet56
        print("state dict length is one of 102, 182, 267, 522")
    except AssertionError as e:
        print("False state dict")

    # indices_list = []
    kept_index_per_layer = {}
    kept_filter_per_layer = {}
    pruned_index_per_layer = {}
    i = 0
    for x in item:
        # print(x[0])
        if 'conv' in x[0] and not 'bias' in x[0]:
            indices_zero, indices_nonzero = check_channel(item[i][1])
            print(x,indices_zero)
            # indices_list.append(indices_nonzero)
            pruned_index_per_layer[item[i][0]] = indices_zero
            kept_index_per_layer[item[i][0]] = indices_nonzero
            kept_filter_per_layer[item[i][0]] = indices_nonzero.shape[0]
        i = i+1

    # add 'module.' if state_dict are store in parallel format
    # state_dict = ['module.' + x for x in state_dict]
   
    if len(item) == 112:
        bottle_block_flag = ['conv_1_3x3.conv.weight',
                             'stage_1.0.conv_a.weight', 'stage_1.0.conv_b.weight',
                             'stage_2.0.conv_a.weight', 'stage_2.0.conv_b.weight',
                             'stage_3.0.conv_a.weight', 'stage_3.0.conv_b.weight']  
        constrct_flag = bottle_block_flag
        block_flag = "conv_b"

    # number of nonzero channel in conv1, and four stages
    num_for_construct = []
    for key in constrct_flag:
        num_for_construct.append(kept_filter_per_layer[key])

    index_for_construct = dict(
        (key, value) for (key, value) in kept_index_per_layer.items() if block_flag in key)
    # bn_value = get_bn_value(big_model, block_flag, pruned_index_per_layer)
    bias_value = get_bias_value(big_model, pruned_index_per_layer)

    if len(item) == 112:
        small_model = models_deploy.resnet56_small(index=kept_index_per_layer, bias_value = bias_value,
                                            num_for_construct=num_for_construct) 

    return kept_index_per_layer, pruned_index_per_layer, block_flag, small_model


def get_bn_value(big_model, block_flag, pruned_index_per_layer):
    big_model.eval()
    bn_flag = "bn3" if block_flag == "conv3" else "bn2"
    key_bn = [x for x in big_model.state_dict().keys() if "bn3" in x] 
    layer_flag_list = [[x[0:6], x[7], x[9:12], x] for x in key_bn if "bias" in x]
    # layer_flag_list = [['layer1', "0", "bn3",'layer1.0.bn3.weight']]
    bn_value = {}

    for layer_flag in layer_flag_list:
        module_bn = big_model._modules.get(layer_flag[0])._modules.get(layer_flag[1])._modules.get(layer_flag[2])
        num_feature = module_bn.num_features
        act_bn = module_bn(Variable(torch.zeros(1, num_feature, 1, 1)))

        index_name = layer_flag[3].replace("bn", "conv").replace("bias", "weight") 
        index = Variable(torch.LongTensor(pruned_index_per_layer[index_name])) 
        act_bn = torch.index_select(act_bn, 1, index) # 

        select = Variable(torch.zeros(1, num_feature, 1, 1))
        select.index_add_(1, index, act_bn)

        bn_value[layer_flag[3]] = select
    return bn_value

def get_bias_value(big_model, pruned_index_per_layer):
    big_model.eval()
    
    key_bias = [x for x in big_model.state_dict().keys() if ".bias" in x and "classifier.bias" not in x]
    layer_flag_list = []
    for x in key_bias:
        if 'conv_1_3x3' in x:
            layer_flag_list.append([x[0:10], x[11:15], x[16:20], x])
        else:
            layer_flag_list.append([x[0:7], x[8], x[10:16], x[17:21], x])

    bias_value = {}

    for layer_flag in layer_flag_list:
        if layer_flag[0] == 'conv_1_3x3':
            module_conv = big_model._modules.get(layer_flag[0])._modules.get(layer_flag[1])
            index_name = layer_flag[3].replace("bias", "weight") 
        else:
            module_conv = big_model._modules.get(layer_flag[0])._modules.get(layer_flag[1])._modules.get(layer_flag[2])
            index_name = layer_flag[4].replace("bias", "weight") 
        
        num_out = module_conv.out_channels
        #act_bn = module_conv2(Variable(torch.zeros(1, num_out, 1, 1)))
        all_bias = module_conv.bias

        index = Variable(torch.LongTensor(pruned_index_per_layer[index_name])) 
        act_bias = torch.index_select(all_bias, 0, index) # 

        select = Variable(torch.zeros(num_out))
        select.index_add_(0, index, act_bias)
        if layer_flag[0] == 'conv_1_3x3':
            bias_value[layer_flag[3]] = select
        else:
            bias_value[layer_flag[4]] = select
    return bias_value
def get_small_model(big_model):
    indice_dict, pruned_index_per_layer, block_flag, small_model = extract_para(big_model)
   
    big_state_dict = big_model.state_dict()
    small_state_dict = {}
    keys_list = list(big_state_dict.keys())
    # print("keys_list", keys_list)
    for index, [key, value] in enumerate(big_state_dict.items()):
        # all the conv layer excluding downsample layer
        flag_conv_ex_down = not 'bias' in key and not 'classifier' in key
        print(flag_conv_ex_down)

        if flag_conv_ex_down:
            small_state_dict[key] = torch.index_select(value, 0, indice_dict[key]) 
            conv_index = keys_list.index(key)

            offset = 1
            bais_key = keys_list[conv_index + offset]
            small_state_dict[bais_key] = torch.index_select(big_state_dict[bais_key], 0, indice_dict[key]) 

        elif 'classifier' in key:
            small_state_dict[key] = value

    small_model.load_state_dict(small_state_dict)
    return small_model


if __name__ == '__main__':
    main()
