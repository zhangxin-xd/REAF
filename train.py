
from tqdm import tqdm
import torch
from collections import OrderedDict
import os, sys
from utils import RecorderMeter, adjust_learning_rate, num_train_examples_per_epoch
from utils import AverageMeter, accuracy, num_val_examples_per_epoch, print_log
import torch.nn.functional as F
import shutil, numpy as np

def train_one_epoch(model, criterion, optimizer, scheduler, train_loader, epoch, args, log, m_mask):  #,
    #current_learning_rate = scheduler.get_lr()[0]
    iters_per_epoch = num_train_examples_per_epoch(args.dataset) // args.batch_size
    losses = AverageMeter()
    loss_weights = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with tqdm(total=iters_per_epoch, desc='training') as (pbar):
        for i, (input, target) in enumerate(train_loader):
            model.train()
            # model_baseline.eval()
            target = target.cuda()
            input = input.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            output = model(input_var)
            # output_baseline = model_baseline(input_var)
            # loss_kd = torch.norm(output-output_baseline) 
            loss = criterion(output, target_var) #+ loss_kd

            n = input.size(0)
            prec1, prec5 = accuracy((output.data), target, topk=(1, 5))
            losses.update(loss.item(), n)
            # loss_weights.update(loss_weight.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            loss.backward(retain_graph=True)
            # loss_weight.backward()
            m_mask.do_grad_mask()
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
            pbar_dic = OrderedDict()
            pbar_dic['epoch'] = epoch
            pbar_dic['cur_iter'] = i
            #pbar_dic['lr'] = current_learning_rate
            pbar_dic['top1'] = '{:.5f}'.format(top1.avg)
            pbar_dic['top5'] = '{:.5f}'.format(top5.avg)
            pbar_dic['loss_c'] = '{:.5f}'.format(losses.avg)
            pbar_dic['loss_w'] = '{:.5f}'.format(loss_weights.avg)
            pbar.set_postfix(pbar_dic)
            pbar.update(1)

    print_log(' **Train** Epoch {epoch:d} Prec@1 {top1:.3f} Prec@5 {top5:.3f} Loss_c@1 {loss_c:.3f} Loss_w@1 {loss_w:.3f}'
            .format(epoch=(epoch), top1=(top1.avg), top5=(top5.avg), loss_c=(losses.avg), loss_w=(loss_weights.avg)), log)
    return (top1.avg, top5.avg, losses.avg)


def val_pruning(model, criterion, test_loader, epoch, args, log):
    model.eval()
    iters_per_epoch_val = num_val_examples_per_epoch(args.dataset) // args.batch_size
    losses_val = AverageMeter()
    top1_val = AverageMeter()
    top5_val = AverageMeter()
    with tqdm(total=iters_per_epoch_val, desc='validating after pruning') as (pbar):
        for i, (input_val, target_val) in enumerate(test_loader):
            target_val = target_val.cuda()
            input_val = input_val.cuda()
            input_var_val = torch.autograd.Variable(input_val)
            target_var_val = torch.autograd.Variable(target_val)
            output_val = model(input_var_val)
            loss_val = criterion(output_val, target_var_val)
            n = input_val.size(0)
            prec1, prec5 = accuracy((output_val.data), target_val, topk=(1, 5))
            losses_val.update(loss_val.item(), n)
            top1_val.update(prec1.item(), n)
            top5_val.update(prec5.item(), n)
            pbar_dic = OrderedDict()
            pbar_dic['epoch'] = epoch
            pbar_dic['cur_iter'] = i
            pbar_dic['top1'] = '{:.5f}'.format(top1_val.avg)
            pbar_dic['top5'] = '{:.5f}'.format(top5_val.avg)
            pbar_dic['loss'] = '{:.5f}'.format(losses_val.avg)
            pbar.set_postfix(pbar_dic)
            pbar.update(1)

    print_log(' **Test** Prec@1 {top1:.3f} Prec@5 {top5:.3f} Error@1 {error1:.3f}'.format(top1=(top1_val.avg), top5=(top5_val.avg), error1=(100 - top1_val.avg)), log)
    return (top1_val.avg, top5_val.avg, losses_val.avg)


def val(model, criterion, test_loader, epoch, args, log):
    model.eval()
    iters_per_epoch_val = int(np.ceil(num_val_examples_per_epoch(args.dataset) / args.batch_size))
    losses_val = AverageMeter()
    top1_val = AverageMeter()
    top5_val = AverageMeter()
    with tqdm(total=iters_per_epoch_val, desc='validating') as (pbar):
        for i, (input_val, target_val) in enumerate(test_loader):
            target_val = target_val.cuda()
            input_val = input_val.cuda()
            input_var_val = torch.autograd.Variable(input_val)
            target_var_val = torch.autograd.Variable(target_val)
            output_val = model(input_var_val)
            loss_val = criterion(output_val, target_var_val)
            n = input_val.size(0)
            prec1, prec5 = accuracy((output_val.data), target_val, topk=(1, 5))
            losses_val.update(loss_val.item(), n)
            top1_val.update(prec1.item(), n)
            top5_val.update(prec5.item(), n)
            pbar_dic = OrderedDict()
            pbar_dic['epoch'] = epoch
            pbar_dic['cur_iter'] = i + 1
            pbar_dic['top1'] = '{:.5f}'.format(top1_val.avg)
            pbar_dic['top5'] = '{:.5f}'.format(top5_val.avg)
            pbar_dic['loss'] = '{:.5f}'.format(losses_val.avg)
            pbar_dic['size'] = input_val.size()
            pbar.set_postfix(pbar_dic)
            pbar.update(1)
    print_log(' **Test** Epoch {epoch:d} Prec@1 {top1:.3f} Prec@5 {top5:.3f} Error@1 {error1:.3f}'.
             format(epoch=(epoch), top1=(top1_val.avg), top5=(top5_val.avg), error1=(100 - top1_val.avg)), log)
    return (top1_val.avg, top5_val.avg, losses_val.avg)


def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)


def train_main(model, criterion, optimizer, train_loader, test_loader, args, log , m_mask):
    recorder = RecorderMeter(args.epoch)
    for epoch in range(args.epoch):
        lr = adjust_learning_rate(optimizer, epoch, args.gammas, args.scheduler, args.base_lr)
        top1, top5, loss = train_one_epoch(model, criterion, optimizer, args.scheduler, train_loader, epoch, args, log, m_mask)
        #if epoch % args.epoch_val == 0:
        # top1_val, top5_val, loss_val = val(model, criterion, test_loader, epoch, args, log)
        # is_best = recorder.update(epoch, loss, top1, loss_val, top1_val)
        # print('learning_rate', lr)

        
        if epoch % args.epoch_pruning == 0 or epoch == args.epoch - 1:
            m_mask.model = model
            m_mask.if_zero()
            m_mask.init_mask(args, epoch)
            m_mask.act_mask()
            m_mask.if_zero()
            model = m_mask.model
            if args.use_cuda:
                model = model.cuda()
        top1_val_pruning, top5_val_pruning, loss_val_pruning = val_pruning(model, criterion, test_loader, epoch, args, log)
        is_best = recorder.update(epoch, loss, top1, loss_val_pruning, top1_val_pruning)

        if is_best:
            print_log(' **BEST** Epoch {epoch:d} Prec@1 {top1:.3f} Prec@5 {top5:.3f} Error@1 {error1:.3f} Lr {lr:.3f}'
                      .format(epoch=(epoch), top1=(top1_val_pruning), top5=(top5_val_pruning), error1=(100 - top1_val_pruning), lr = lr), log)
            save_checkpoint({'epoch':epoch + 1,  'arch':args.arch, 
                'state_dict':model.state_dict(), 
                'optimizer':optimizer}, is_best, args.weight_path, 'best' + '.pth')
        if epoch == (args.epoch - 1):
            print_log(' **Final** Epoch {epoch:d} Prec@1 {top1:.3f} Prec@5 {top5:.3f} Error@1 {error1:.3f} Lr {lr:.3f}'
                      .format(epoch=(epoch), top1=(top1_val_pruning), top5=(top5_val_pruning), error1=(100 - top1_val_pruning), lr = lr), log)
            save_checkpoint({'epoch':epoch + 1,  'arch':args.arch, 
                'state_dict':model.state_dict(), 
                'optimizer':optimizer}, is_best, args.weight_path, 'final' + '.pth')
        # if epoch % args.epoch_save == 0 or epoch == args.epoch - 1:
        #     print('saving model')
        #     save_checkpoint({'epoch':epoch + 1,  'arch':args.arch, 
        #         'state_dict':model.state_dict(), 
        #         'optimizer':optimizer}, is_best, args.weight_path, 'checkpoint_' + str(epoch + 1) + '.pth')