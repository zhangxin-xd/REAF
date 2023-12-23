import numpy as np
import torch
from scipy.spatial import distance
class Mask:
    def __init__(self, model):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.distance_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        self.mask_ori = []
        self.mask_shadow = []
        self.filter_small_index = {}
        self.filter_index = {}
        self.filter_large_index = {}
        self.similar_matrix = {}
        self.norm_matrix = {}

    def get_codebook(self, weight_torch, compress_rate, length):
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()

        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)

        threshold = weight_sort[int(length * (1 - compress_rate))]
        weight_np[weight_np <= -threshold] = 1
        weight_np[weight_np >= threshold] = 1
        weight_np[weight_np != 1] = 0

        print("codebook done")
        return weight_np

    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0
        else:
            pass
        return codebook, filter_index

    def get_filter_index(self, weight_torch, compress_rate, length):
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            filter_small_index = []
            filter_large_index = []
            filter_large_index = norm2_np.argsort()[filter_pruned_num:]
            filter_small_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            # print("filter index done")
        else:
            pass
        return filter_small_index, filter_large_index

    # optimize for fast ccalculation
    def get_filter_similar(self, weight_torch, compress_rate, length):
        codebook = np.ones(length[0])
        if (len(weight_torch[0][2].size()) == 4) and (len(weight_torch[1][2].size()) == 4):
            filter_pruned_num = int(weight_torch[0][2].size()[0] * (1 - compress_rate)) # 要剪掉的filter数
            filter_maintain_num = int(weight_torch[0][2].size()[0] - filter_pruned_num)
            weight_vec_ori = weight_torch[0][2].view(weight_torch[0][2].size()[0], -1).cpu().detach().numpy()
            weight_vec_shadow = weight_torch[1][2].view(weight_torch[1][2].size()[0], -1).cpu().detach().numpy()
            # ori
            similar_matrix = distance.cdist(weight_vec_ori, weight_vec_ori, 'euclidean')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)
            similar_max_add_min = similar_sum.max() + similar_sum.min()
            similar_max_minu_min = similar_sum.max() - similar_sum.min()
            similar = (torch.tensor((-2*similar_sum+similar_max_add_min)/similar_max_minu_min)).cuda()# similar_sum larger, similar smaller
            similar_sig = torch.sigmoid(similar) #similar smaller,  similar_sig smaller
            # ori shadow 
            similar_matrix_so = distance.cdist(weight_vec_ori, weight_vec_shadow, 'euclidean')
            similar_sum_so = np.sum(np.abs(similar_matrix_so), axis=0)
            similar_sum_sum_so = np.sum(np.abs(similar_sum_so), axis=0)
            p = torch.tensor((similar_sum_so/similar_sum_sum_so)).cuda()
            Entropy = -p * torch.log2(p)  
            Entropy_max_add_min = Entropy.max() + Entropy.min()
            Entropy_max_minu_min = Entropy.max() - Entropy.min()
            Entropy_1 = (-2*Entropy+Entropy_max_add_min)/Entropy_max_minu_min  # Entropy larger, Entropy_1 smaller
            Entropy_sig = torch.sigmoid(Entropy_1)
            
            similar_Entropy = similar_sig * Entropy_sig

            index_test = similar_sig.argsort()[:filter_maintain_num]
            filter_pruned_index = similar_Entropy.argsort()[filter_maintain_num:] # 
            filter_maintain_index = similar_Entropy.argsort()[:filter_maintain_num]

            print('filter_pruned_index', filter_pruned_index)
            print('filter_maintain_index', filter_maintain_index)
           

            kernel_length = weight_torch[0][2].size()[1] * weight_torch[0][2].size()[2] * weight_torch[0][2].size()[3]
            for x in range(0, len(filter_pruned_index)):
                codebook[
                filter_pruned_index[x] * kernel_length: (filter_pruned_index[x] + 1) * kernel_length] = 0
            print("similar index done")
        else:
            pass
        return codebook, filter_pruned_index
      
    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x
    
    # 计算每层参数tensor形状\参数量
    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]

    def init_rate(self, args, epoch):
        for index, item in enumerate(self.model.parameters()):
            self.compress_rate[index] = 1
        for key in range(args.layer_begin, args.layer_end + 1, args.layer_inter):
            self.compress_rate[key] = self.cal_rate(args,epoch) #设置每层的剪枝率
        # different setting for  different architecture
        if args.arch == 'src56':
            last_index = 330
            skip_list = [] # skip projection
            self.mask_shadow = [x for x in range(3, last_index, 6)] # 所有conv
        elif args.arch == 'src110':
            last_index = 654
            skip_list = [] # skip projection
            self.mask_shadow = [x for x in range(3, last_index, 6)] # 所有conv
        # to jump the last fc layer 只取conv跳过bn和fc
        self.mask_index = [x for x in range(0, last_index, 3)] # 所有conv
        
        # 跳过projection
        for x in skip_list:
            self.compress_rate[x] = 1
            self.mask_index.remove(x)
        print(self.mask_index) 
        
        for x in self.mask_shadow:
            self.mask_index.remove(x)
        print(self.mask_index)
        # 此时的self.mask_index不含projection、shadow,只有主干的conv index
        self.mask_ori = [x-3 for x in self.mask_shadow]
 

    def init_mask(self, args,epoch):
        self.init_rate(args,epoch)
        ori_shadow_layer = []
        model_length = []
        for index, (name, para) in enumerate(self.model.named_parameters()):
            if (index in self.mask_ori) or (index in self.mask_shadow):
                ori_shadow_layer.append([index, name, para])
                model_length.append(self.model_length[index])                
                if len(ori_shadow_layer) == 2:
                    self.mat[index-3], self.filter_index[index-3] = self.get_filter_similar(ori_shadow_layer, self.compress_rate[index],
                                                             model_length)
                    self.mat[index-3] = self.convert2tensor(self.mat[index-3])
                    self.mat[index] = self.convert2tensor(self.mat[index-3])
                    if args.use_cuda:
                        self.mat[index-3] = self.mat[index-3].cuda()
                        self.mat[index] = self.mat[index].cuda()
                    ori_shadow_layer = []
                    model_length = []
            else:
                pass
        print("mask Ready")

    def cal_rate(self, args, epoch):
        if args.dataset == 'cifar100' or args.dataset == 'cifar10':
            if epoch<=75:
                if args.pruning_rate == 0.5:
                    lam = 0.403
                if args.pruning_rate == 0.6:
                    lam = 0.322
                if args.pruning_rate == 0.4:
                    lam = 0.484    
                if args.pruning_rate == 0.3:
                    lam = 0.5641        
                if args.pruning_rate == 0.7:
                    lam = 0.24175     
                rate = 1- lam*epoch**0.05 #0.403 for 0.5  0.322 for 0.6 0.484 for 0.4
            else:
                rate = args.pruning_rate
        if args.dataset == 'imagenet_standard':
            if epoch<=30:
                if args.pruning_rate == 0.5:
                    lam = 0.4218
                if args.pruning_rate == 0.6:
                    lam = 0.4715
                if args.pruning_rate == 0.4:
                    lam = 0.5062             
                rate = 1- lam*epoch**0.05 #0.403 for 0.5  0.322 for 0.6 0.484 for 0.4
            else:
                rate = args.pruning_rate
        return rate
    def act_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
            if index in self.mask_shadow:
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
        print("mask Done")

    def do_grad_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.grad.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.grad.data = b.view(self.model_size[index])
            if index in self.mask_shadow:
                a = item.grad.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.grad.data = b.view(self.model_size[index])
        # print("grad zero Done")

    def if_zero(self):
        for index, item in enumerate(self.model.parameters()):
            if (index in self.mask_index):
                # if index == 0:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()

                print(
                    "number of nonzero weight is %d, zero is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))
