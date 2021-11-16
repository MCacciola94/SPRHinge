"""
Group Sparsity: The Hinge Between Filter Pruning and Decomposition forNetwork Compression
This module implement the proposed Hinge method to ResNet20 and ResNet56.
This is the simplified version of hinge_resnet_basic_complex.py. A lot of unused functions and procedures are deleted.
"""
__author__ = 'Yawei Li'
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from easydict import EasyDict as edict
from resnet import ResNet, ResBlock
from hinge_utility import init_weight_proj, get_nonzero_index, plot_figure, plot_per_layer_compression_ratio
from flops_counter import get_model_complexity_info
from torch.nn.utils import prune

#My pkgs
import aux_tools as at

########################################################################################################################
# used to edit the modules and parameters during the PG optimization and continuing training stage.
########################################################################################################################

def get_compress_idx(module, percentage, threshold):
    body = module._modules['body']
    conv12 = body._modules['0']._modules['1']
    conv22 = body._modules['3']._modules['1']
    projection1 = conv12.weight.data.squeeze().t()
    projection2 = conv22.weight.data.squeeze().t()
    norm1, pindex1 = get_nonzero_index(projection1, dim='output', counter=1, percentage=percentage, threshold=threshold)
    fix_channel =  0
    norm2, pindex2 = get_nonzero_index(projection2, dim='input', counter=1, percentage=percentage, threshold=threshold,
                                       fix_channel=fix_channel)
    def _get_compress_statistics(norm, pindex):
        remain_norm = norm[pindex]
        channels = norm.shape[0]
        remain_channels = remain_norm.shape[0]
        remain_norm = remain_norm.detach().cpu()
        stat_channel = [channels, channels - remain_channels, (channels - remain_channels) / channels]
        stat_remain_norm = [remain_norm.max(), remain_norm.mean(), remain_norm.min()]
        return edict({'stat_channel': stat_channel, 'stat_remain_norm': stat_remain_norm,
                      'remain_norm': remain_norm, 'pindex': pindex})

    return [_get_compress_statistics(norm1, pindex1), _get_compress_statistics(norm2, pindex2)]


# ========
# Double ReLU activation function
# ========
class DReLU(nn.Module):
    def __init__(self):
        super(DReLU, self).__init__()
        self.slope_p = nn.Parameter(torch.tensor(1.0))
        self.slope_n = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        out = self.slope_p * F.relu(x) + self.slope_n * F.relu(-x)
        return out

# ========
# Modify submodules
# ========
def modify_submodules(module):
    conv1 = [module._modules['body']._modules['0'], nn.Conv2d(10, 10, 1, bias=False)]
    module._modules['body']._modules['0'] = nn.Sequential(*conv1)
    module._modules['body']._modules['2'] = DReLU()
    conv2 = [module._modules['body']._modules['3'], nn.Conv2d(10, 10, 1, bias=False)]
    module._modules['body']._modules['3'] = nn.Sequential(*conv2)
    module.optimization = True



# ========
# Set submodule parameters
# ========
def set_module_param(module, params):

    ws1 = params.weight1.size()
    ps1 = params.projection1.size()
    ws2 = params.weight2.size()
    ps2 = params.projection2.size()

    body = module._modules['body']
    conv11 = body._modules['0']._modules['0']
    conv12 = body._modules['0']._modules['1']
    conv21 = body._modules['3']._modules['0']
    conv22 = body._modules['3']._modules['1']

    # set conv11
    conv11.in_channels = ws1[1]
    conv11.out_channels = ws1[0]
    conv11.weight.data = params.weight1.data
    conv11.bias = None

    # set conv12
    conv12 = nn.Conv2d(ps1[1],ps1[0],1,bias = False)
    #conv12.in_channels = ps1[1]
    #conv12.out_channels = ps1[0]

    conv12.weight.data[:]=params.projection1[:]

    
    if params.bias1 is not None:
        conv12.bias = nn.Parameter(params.bias1)
    # Note that the bias term is added to the second conv
    # Do not need to set batchnorm1, activation, and batchnorm2.

    # set conv11
    conv21.in_channels = ws2[1]
    conv21.out_channels = ws2[0]
    conv21.weight.data = params.weight2.data
    conv21.bias = None

    # set conv12
    # conv22.in_channels = ps2[1]
    # conv22.out_channels = ps2[0]
    conv22 = nn.Conv2d(ps2[1],ps2[0],1,bias = False)
    conv22.weight.data[:] = params.projection2.data[:]
    
    body._modules['0']._modules['1']=conv12.cuda()
    body._modules['3']._modules['1']=conv22.cuda()
    if params.bias2 is not None:
        conv22.bias = nn.Parameter(params.bias2)
    
    

# ========
# Compress module parameters
# ========
def compress_module_param(module, percentage, threshold):
    body = module._modules['body']
    conv11 = body._modules['0']._modules['0']
    conv12 = body._modules['0']._modules['1']
    batchnorm1 = body._modules['1']
    conv21 = body._modules['3']._modules['0']
    conv22 = body._modules['3']._modules['1']
    

    ws1 = conv11.weight.shape
    
    projection1 = conv12.weight.data.squeeze().t()
    bias1 = conv12.bias.data if conv12.bias is not None else None
    bn_weight1 = batchnorm1.weight.data
    bn_bias1 = batchnorm1.bias.data
    bn_mean1 = batchnorm1.running_mean.data
    bn_var1 = batchnorm1.running_var.data

    ws2 = conv21.weight.shape
    weight2 = conv21.weight.data.view(ws2[0], -1).t()
    projection2 = conv22.weight.data.squeeze().t()


    _, pindex1 = get_nonzero_index(projection1, dim='output', counter=1, percentage=percentage, threshold=threshold)
    fix_channel =  0
    _, pindex2 = get_nonzero_index(projection2, dim='input', counter=1, percentage=percentage, threshold=threshold,
                                   fix_channel=fix_channel)
   

    # conv11 don't need to be changed.
    # compress conv12: projection1, bias1
    projection1 = torch.index_select(projection1, dim=1, index=pindex1)
    conv12.weight = nn.Parameter(projection1.t().view(pindex1.shape[0], ws1[0], 1, 1)) #TODO: check this one.
    if bias1 is not None:
        conv12.bias = nn.Parameter(torch.index_select(bias1, dim=0, index=pindex1))
    conv12.out_channels = conv12.weight.size()[0]

    # compress batchnorm1
    batchnorm1.weight = nn.Parameter(torch.index_select(bn_weight1, dim=0, index=pindex1))
    batchnorm1.bias = nn.Parameter(torch.index_select(bn_bias1, dim=0, index=pindex1))
    batchnorm1.running_mean = torch.index_select(bn_mean1, dim=0, index=pindex1)
    batchnorm1.running_var = torch.index_select(bn_var1, dim=0, index=pindex1)
    batchnorm1.num_features = len(batchnorm1.weight)
    index = torch.repeat_interleave(pindex1, ws2[2] * ws2[3]) * ws2[2] * ws2[3] \
            + torch.tensor(range(0, ws2[2] * ws2[3])).repeat(pindex1.shape[0]).cuda()
    # compress conv21
    weight2 = torch.index_select(weight2, dim=0, index=index)
    weight2 = torch.index_select(weight2, dim=1, index=pindex2)
    conv21.weight = nn.Parameter(weight2.t().view(pindex2.shape[0], pindex1.shape[0], ws2[2], ws2[3]))
    conv21.out_channels, conv21.in_channels = conv21.weight.size()[:2]
    # compress conv22
    projection2 = torch.index_select(projection2, dim=0, index=pindex2)
    conv22.weight = nn.Parameter(projection2.t().view(-1, pindex2.shape[0], 1, 1))
    conv22.in_channels = conv22.weight.size()[1]
    # bias2 don't need to be changed.
 

def modify_network(net_current):
    args = net_current.args
    modules = []
    for module_cur in net_current.modules():
        if isinstance(module_cur, ResBlock):
            modules.append(module_cur)
    for module_cur in modules:

        # get initialization values for the ResBlock to be compressed
        weight1 = module_cur.state_dict()['body.0.weight']
        weight2 = module_cur.state_dict()['body.3.weight']
        if args.init_method.find('disturbance') >= 0:
            weight1, projection1 = init_weight_proj(weight1, init_method=args.init_method, d=0, s=0.05)
            weight2, projection2 = init_weight_proj(weight2, init_method=args.init_method, d=1, s=0.05)
        else:
            weight1, projection1 = init_weight_proj(weight1, init_method=args.init_method)
            weight2, projection2 = init_weight_proj(weight2, init_method=args.init_method)
        # modify submodules in the ResBlock
        modify_submodules(module_cur)
        # set ResBlock module params
        params = edict({'weight1': weight1, 'projection1': projection1, 'bias1': None,
                        'weight2': weight2, 'projection2': projection2, 'bias2': None})
        set_module_param(module_cur, params)


def make_model(args):
    return Hinge(args)


class Hinge(ResNet):

    def __init__(self, args):
        self.args = args
   
        super(Hinge, self).__init__(self.args)


        self.input_dim = (3, 32, 32)

        self.flops, self.params = get_model_complexity_info(self, self.input_dim, print_per_layer_stat=False)

        self.register_buffer('running_grad_ratio', None)

        modify_network(self)

    def find_modules(self):
        return [m for m in self.modules() if isinstance(m, ResBlock)]

    def sparse_param(self, module):
        param1 = module.state_dict(keep_vars=True)['body.0.1.weight']
        param2 = module.state_dict(keep_vars=True)['body.3.1.weight']
        return param1, param2


    def merge_conv(self):
        # used after finetuning to save the merged model
        for m in self.find_modules():
            body = m._modules['body']
            convs = body._modules['0']
            if isinstance(convs, nn.Sequential):
                ws = convs._modules['0'].weight.size()
                ps = convs._modules['1'].weight.size()
                weight = convs._modules['0'].weight.data.view(ws[0], -1).t()
                projection = convs._modules['1'].weight.data.squeeze().t()
                weight = torch.mm(weight, projection).t().view(ps[0], ws[1], ws[2], ws[3])
                body._modules['0'] = convs._modules['0']
                body._modules['0'].weight = nn.Parameter(weight)
                body._modules['0'].out_channels = ps[0]

    def split_conv(self, state_dict):
        for m in self.find_modules():
            body = m._modules['body']
            conv1 = [body._modules['0'], nn.Conv2d(10, 10, 1, bias=False)]
            body._modules['0'] = nn.Sequential(*conv1)
        self.load_state_dict(state_dict, strict=False)

    def set_channels(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.out_channels, m.in_channels = m.weight.size()[:2]
            elif isinstance(m, nn.BatchNorm2d):
                m.num_features = m.weight.size()[0]
          


