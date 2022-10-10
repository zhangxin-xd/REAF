import torch
def sgd_optimizer(args, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.base_lr
        weight_decay = args.weight_decay
        if "bias" in key or "bn" in key or "BN" in key:
            # lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = args.weight_decay
            print('set weight_decay={} for {}'.format(weight_decay, key))
        if 'bias' in key:
            apply_lr = 2 * lr
            print('set lr={} for {}'.format(apply_lr, key))
        else:
            apply_lr = lr

        params += [{"params": [value], "lr": apply_lr, "weight_decay": weight_decay}]
    # optimizer = torch.optim.Adam(params, lr)
    optimizer = torch.optim.SGD(params, lr, momentum=args.momentum)
    return optimizer

def get_optimizer(args, model):
    return sgd_optimizer(args, model)
