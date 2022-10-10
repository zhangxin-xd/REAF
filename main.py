
from genericpath import exists
import os, random, sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from collections import namedtuple
import numpy as np
from  creaters.creater import ConvCreater
from creaters.shadow_creater import ShadowCreater
from model_cfg import get_model_fn
from optimizer import get_optimizer
from do_mask import Mask
from train import train_main, val_pruning, val
from utils import print_log

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='', help='Path to dataset')
parser.add_argument('--weight_path', type=str, default='', help='Path to weight')
parser.add_argument('--dataset', type=str, default='cifar10', help='Choose from cifar10 and cifar 100')
parser.add_argument('--epoch', type=int, default=300, help='Epoch')
parser.add_argument('--epoch_val', type=int, default=60, help='Epoch for val')
parser.add_argument('--epoch_pruning', type=int, default=1, help='Epoch for val')
parser.add_argument('--epoch_save', type=int, default=100, help='Epoch for save')
parser.add_argument('--batch_size', type=int, default=256, help='Batchsize')
parser.add_argument('--resume', type=str, default='', help='pretrain model')
parser.add_argument('--base_lr', type=float, default=0.01, help='Learning_rate') 
parser.add_argument('--gammas', type=list, default=[0.1, 0.1], help='Learning_rate') 
parser.add_argument('--scheduler', type=list, default=[150, 225], help='Learning_rate') 

parser.add_argument('--cosine_minimum', type=float, default=0, help='Learning_rate') 
parser.add_argument('--linear_final_lr', type=float, default=None, help='Learning_rate') 
parser.add_argument('--lr_epoch_boundaries', type=int, default=None, help='Learning_rate') 
parser.add_argument('--warmup_factor', type=float, default=0, help='Learning_rate') 
parser.add_argument('--warmup_epochs', type=int, default=0, help='Learning_rate')  
parser.add_argument('--warmup_method', type=str, default='linear', help='Learning_rate') 
parser.add_argument('--lr_decay_factor', type=float, default=None, help='Learning_rate') 


parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--weight_decay', type=float, nargs='+', default=5e-4, help='Weight_decay')  


parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default='0', help='Number of workers')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--arch', type=str, default='src56', help='Choose from ')
parser.add_argument('--block_type', type=str, default='shadow', help='Reparameterized or not, shadow is yes, normal is no')
parser.add_argument('--af', type=bool, default=True)
parser.add_argument('--pruning_rate', type=float, default=0.6, help='remaining rate')
parser.add_argument('--layer_begin', type=int, default=0, help='Start layer')
parser.add_argument('--layer_end', type=int, default=0, help='End layer')
parser.add_argument('--layer_inter', type=int, default=3, help='Interval Layer')


args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

if args.manualSeed is None: 
    args.manualSeed = random.randint(1, 10000) 
random.seed(args.manualSeed) 
torch.manual_seed(args.manualSeed) 
if args.use_cuda: 
    torch.cuda.manual_seed_all(args.manualSeed) 
cudnn.benchmark = True 

if __name__ == '__main__':

    network_type = args.arch 
    block_type = args.block_type
    assert block_type in ['shadow', 'normal']

    ###########################################################################    
    ## prepare dataset

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
        train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 100
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    ###############################################################################
    ## define network 
    gamma_init = None

    if args.arch == 'src56':
        weight_decay = 5e-4
        #   ------------------------------------
        #   94.47  --->  95+
        #batch_size = 128
        warmup_epochs = 0
        gamma_init = 0.5
        args.layer_begin = 0
        args.layer_end = 327 # 330-1-3
        args.layer_inter = 3
    if args.arch == 'src110':
        weight_decay = 5e-4
        #   ------------------------------------
        #   94.47  --->  95+
        #batch_size = 128
        warmup_epochs = 0
        gamma_init = 0.5
        args.layer_begin = 0
        args.layer_end = 651 # 330-1-3
        args.layer_inter = 3
    # Remembering enhancement
    if block_type == 'shadow':
        creater = ShadowCreater(deploy=False, gamma_init=gamma_init)
    else:
        creater = ConvCreater()
    
    net = get_model_fn(args.dataset, args.arch)
    print(net)
    model = net(creater)
    print(model)
    #model = torch.nn.DataParallel(model, device_ids= list(args.gpus))
    model = model.cuda()
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
    ##Init criterion, optimizer, scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(args, model)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
    #scheduler= get_lr_scheduler(args, optimizer)
    ###############################################################################
    ## 
    m = Mask(model)
    m.init_length()
    print('-'*10+'pruning begin' + '-'*10)
    print('pruning rate is %f'% args.pruning_rate)
    m.model = model
    model = m.model
    if args.use_cuda:
       model = model.cuda()
    ###############################################################################
    ## begin training
    if not os.path.isdir(args.weight_path): 
        os.makedirs(args.weight_path) 
    log = open(os.path.join(args.weight_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w') 
    print_log('save path : {}'.format(args.weight_path), log) 
    state = {k: v for k, v in args._get_kwargs()} 
    print_log(state, log) 
    print_log("Random Seed: {}".format(args.manualSeed), log) 
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log) 
    print_log("torch version : {}".format(torch.__version__), log) 
    print_log("cudnn version : {}".format(torch.backends.cudnn.version()), log) 
    print_log("Pruning Rate: {}".format(args.pruning_rate), log) 
    print_log("Layer Begin: {}".format(args.layer_begin), log) 
    print_log("Layer End: {}".format(args.layer_end), log) 
    print_log("Layer Inter: {}".format(args.layer_inter), log) 
    print_log("Epoch prune: {}".format(args.epoch_pruning), log) 


    train_main(model=model, criterion = criterion, optimizer = optimizer, train_loader = train_loader, 
                    test_loader=test_loader, args = args, log = log, m_mask = m) #
        # val(model = model, criterion = criterion, test_loader = test_loader, epoch = 0, args = args)
    ################################################################################
    
       

