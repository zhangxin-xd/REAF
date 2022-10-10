
from os import remove
import torch 
import numpy as np
from model_cfg import get_model_fn
from creaters.shadow_creater import ShadowCreater
from collections import OrderedDict
SQUARE_KERNEL_KEYWORD = 'square_conv.weight'

def _fuse_kernel(kernel, gamma, std):
    b_gamma = torch.reshape(gamma, (kernel.shape[0], 1, 1, 1))
    b_gamma = b_gamma.repeat(1, kernel.shape[1], kernel.shape[2], kernel.shape[3])
    b_std = torch.reshape(std, (kernel.shape[0], 1, 1, 1))
    b_std = b_std.repeat(1, kernel.shape[1], kernel.shape[2], kernel.shape[3])
    return kernel * b_gamma / b_std

def _add_to_square_kernel(square_kernel, asym_kernel):
    asym_h = asym_kernel.shape[2]
    asym_w = asym_kernel.shape[3]
    square_h = square_kernel.shape[2]
    square_w = square_kernel.shape[3]
    square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
                                        square_w // 2 - asym_w // 2 : square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

# 
def convert_weights(train_weights, deploy_weights, eps):
    train_dict = torch.load(train_weights)['state_dict'] 
    
    #print(train_dict.keys())
    deploy_dict = {}
    square_conv_var_names = [name for name in train_dict if SQUARE_KERNEL_KEYWORD in name]
    square_conv_var_names = [name for name in train_dict.keys() if SQUARE_KERNEL_KEYWORD in name] 
    for square_name in square_conv_var_names:
        square_kernel = train_dict[square_name]
        square_mean = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'square_bn.running_mean')]
        square_std = torch.sqrt(train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'square_bn.running_var')] + eps)
        square_gamma = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'square_bn.weight')]
        square_beta = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'square_bn.bias')]

        shadow_kernel = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'shadow_conv.weight')]
        shadow_mean = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'shadow_bn.running_mean')]
        shadow_std = torch.sqrt(train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'shadow_bn.running_var')] + eps)
        shadow_gamma = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'shadow_bn.weight')]
        shadow_beta = train_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'shadow_bn.bias')]


        fused_bias = square_beta + shadow_beta - square_mean * square_gamma / square_std \
                    - shadow_mean * shadow_gamma / shadow_std
        fused_kernel = _fuse_kernel(square_kernel, square_gamma, square_std)

        _add_to_square_kernel(fused_kernel, _fuse_kernel(shadow_kernel, shadow_gamma, shadow_std))
       

        deploy_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'fused_conv.weight')] = fused_kernel
        deploy_dict[square_name.replace(SQUARE_KERNEL_KEYWORD, 'fused_conv.bias')] = fused_bias

    for k, v in train_dict.items():
        if 'shadow_' not in k and 'square_' not in k:
            deploy_dict[k] = v
    torch.save(deploy_dict, deploy_weights)
def remove_module_dict(state_dict): 
    new_state_dict = OrderedDict() 
    for k, v in state_dict.items(): 
        name = name[7:] if name.startswith("module.") else name # remove `module.` 
        new_state_dict[name] = v 
    return new_state_dict 
if __name__ == '__main__':
    N = 1
    C = 3
    H = 200
    W = 200
    O = 8


    x = torch.randn(128, 3, 32, 32)*10
    print('input shape is ', x.size()) 
    target_weights = '' 

    net = get_model_fn('cifar10', 'src56')
    creater = ShadowCreater(deploy=False, gamma_init=0.5)
    Network = net(creater)
    checkpoint = torch.load(target_weights)
    state_dict = checkpoint['state_dict']
    Network.load_state_dict(state_dict)
    Network.eval()
    output_shadow = Network(x)


    convert_weights(target_weights, target_weights.replace('.pth', '_deploy.pth'), eps=1e-5)#转化网络
    checkpoint_fusion = torch.load(target_weights.replace('.pth', '_deploy.pth'))
    creater = ShadowCreater(deploy=True)
    Network_fusion = net(creater)
    state_dict_fusion = checkpoint_fusion
    Network_fusion.load_state_dict(state_dict_fusion)
    Network_fusion.eval()
    output_fusion = Network_fusion(x)
 
 
    rediuse = output_fusion-output_shadow
    print(output_fusion-output_shadow)


