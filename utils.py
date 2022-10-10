import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os, sys, time
import random

def adjust_learning_rate(optimizer, epoch, gammas, scheduler, learning_rate): 
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs""" 
    lr = learning_rate
    assert len(gammas) == len(scheduler), "length of gammas and scheduler should be equal" 
    for (gamma, step) in zip(gammas, scheduler): 
        if (epoch >= step): 
            lr = lr * gamma 
        else: 
            break 
    for param_group in optimizer.param_groups: 
        param_group['lr'] = lr 
    return lr 
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

def num_train_examples_per_epoch(dataset_name):
    if 'imagenet' in dataset_name:
        return 13000#1281167
    elif dataset_name == 'nwpu-45':
        return 25200
    elif dataset_name == 'cifar100':
        return 50000
    elif dataset_name == 'cifar10':
        return 50000
    else:
        assert False
def num_val_examples_per_epoch(dataset_name):
    if 'imagenet' in dataset_name:
        return 50000
    elif dataset_name in ['nwpu-45', 'ch']:
        return 6300
    elif dataset_name == 'cifar100':
        return 10000
    elif dataset_name == 'cifar10':
        return 10000
    else:
        assert False

def accuracy(output, target, topk=(1,)): 
    """Computes the precision@k for the specified values of k""" 
    maxk = max(topk) 
    batch_size = target.size(0) 
    _, pred = output.topk(maxk, 1, True, True) 
    pred = pred.t() 
    correct = pred.eq(target.view(1, -1).expand_as(pred)) 
    res = [] 
    for k in topk: 
        correct_k = correct[:k].contiguous().view(-1).float().sum(0) 
        res.append(correct_k.mul_(100.0 / batch_size)) 
    return res 


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write("{}\n".format(print_string))
    log.flush()

class RecorderMeter(object):
  """Computes and stores the minimum loss value and its epoch index"""
  def __init__(self, total_epoch):
    self.reset(total_epoch)

  def reset(self, total_epoch):
    assert total_epoch > 0
    self.total_epoch   = total_epoch
    self.current_epoch = 0
    self.epoch_losses  = np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_losses  = self.epoch_losses - 1

    self.epoch_accuracy= np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_accuracy= self.epoch_accuracy

  def update(self, idx, train_loss, train_acc, val_loss, val_acc):
    assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
    self.epoch_losses  [idx, 0] = train_loss
    self.epoch_losses  [idx, 1] = val_loss
    self.epoch_accuracy[idx, 0] = train_acc
    self.epoch_accuracy[idx, 1] = val_acc
    self.current_epoch = idx + 1
    return self.max_accuracy(False) == np.array(val_acc).astype(np.float32)

  def max_accuracy(self, istrain):
    if self.current_epoch <= 0: return 0
    if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
    else:       return self.epoch_accuracy[:self.current_epoch, 1].max()
  
  def plot_curve(self, save_path):
    title = 'the accuracy/loss curve of train/val'
    dpi = 80  
    width, height = 1200, 800
    legend_fontsize = 10
    scale_distance = 48.8
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    x_axis = np.array([i for i in range(self.total_epoch)]) # epochs
    y_axis = np.zeros(self.total_epoch)

    plt.xlim(0, self.total_epoch)
    plt.ylim(0, 100)
    interval_y = 5
    interval_x = 5
    plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
    plt.yticks(np.arange(0, 100 + interval_y, interval_y))
    plt.grid()
    plt.title(title, fontsize=20)
    plt.xlabel('the training epoch', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
  
    y_axis[:] = self.epoch_accuracy[:, 0]
    plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_accuracy[:, 1]
    plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    
    y_axis[:] = self.epoch_losses[:, 0]
    plt.plot(x_axis, y_axis*50, color='g', linestyle=':', label='train-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_losses[:, 1]
    plt.plot(x_axis, y_axis*50, color='y', linestyle=':', label='valid-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    if save_path is not None:
      fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
      print ('---- save figure {} into {}'.format(title, save_path))
    plt.close(fig)
def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def convert_secs2time(epoch_time):
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  return need_hour, need_mins, need_secs

def time_file_str():
  ISOTIMEFORMAT='%Y-%m-%d'
  string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string + '-{}'.format(random.randint(1, 10000))

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print ('%s function took %0.3f ms' % (f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap