"""
Group Sparsity: The Hinge Between Filter Pruning and Decomposition forNetwork Compression
This module implement the proposed Hinge method to ResNet20 and ResNet56.
"""
__author__ = 'Yawei Li'
import torch
import torch.nn as nn
import os
import math
from easydict import EasyDict as edict
from resnet import ResBlock, ResNet
from hinge_utility import init_weight_proj, get_nonzero_index, plot_figure, plot_per_layer_compression_ratio
from flops_counter import get_model_complexity_info
from hinge_resnet_basic import modify_submodules, set_module_param
#from IPython import embed
from spr_reg import spr_comp


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
    norm1, pindex1 = get_nonzero_index(projection1, dim='input', counter=1, percentage=percentage, threshold=threshold)
    fix_channel = 0
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
# Compress module parameters
# ========
def compress_module_param(module, percentage, threshold):
    body = module._modules['body']
    conv11 = body._modules['0']._modules['0']
    conv12 = body._modules['0']._modules['1']
    conv21 = body._modules['3']._modules['0']
    conv22 = body._modules['3']._modules['1']

    ws1 = conv11.weight.shape
    weight1 = conv11.weight.data.view(ws1[0], -1).t()
    projection1 = conv12.weight.data.squeeze().t()

    ws2 = conv21.weight.shape
    weight2 = conv21.weight.data.view(ws2[0], -1).t()
    projection2 = conv22.weight.data.squeeze().t()

    _, pindex1 = get_nonzero_index(projection1, dim='input', counter=1, percentage=percentage, threshold=threshold)
    fix_channel = 0
    _, pindex2 = get_nonzero_index(projection2, dim='input', counter=1, percentage=percentage, threshold=threshold,
                                   fix_channel=fix_channel)

    # compress conv11.
    weight1 = torch.index_select(weight1, dim=1, index=pindex1)
    conv11.weight = nn.Parameter(weight1.t().view(pindex1.shape[0], ws1[1], ws1[2], ws1[3]))
    conv11.out_channels, conv11.in_channels = conv11.weight.size()[:2]
    # compress conv12: projection1, bias1
    projection1 = torch.index_select(projection1, dim=0, index=pindex1)
    conv12.weight = nn.Parameter(projection1.t().view(ws1[0], pindex1.shape[0], 1, 1))
    conv12.out_channels, conv12.in_channels = conv12.weight.size()[:2]

    # compress conv21
    weight2 = torch.index_select(weight2, dim=1, index=pindex2)
    conv21.weight = nn.Parameter(weight2.t().view(pindex2.shape[0], ws2[1], ws2[2], ws2[3]))
    conv21.out_channels, conv21.in_channels = conv21.weight.size()[:2]
    # compress conv22
    projection2 = torch.index_select(projection2, dim=0, index=pindex2)
    conv22.weight = nn.Parameter(projection2.t().view(ws2[0], pindex2.shape[0], 1, 1))
    conv22.out_channels, conv22.in_channels = conv22.weight.size()[:2]


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
            weight1, projection1 = init_weight_proj(weight1, init_method=args.init_method, d=1, s=0.05)
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
    
    print("111111111111111111111")
    

def make_model(args):
    return Hinge(args)
   

class Hinge(ResNet):

    def __init__(self, args):
        self.args = args
        super(Hinge, self).__init__(self.args)

        
        self.input_dim = (3, 32, 32)
       
        self.flops, self.params = get_model_complexity_info(self, self.input_dim, print_per_layer_stat=False)

        # if self.args.model.lower().find('resnet') >= 0:
        # self.register_parameter('running_grad_ratio', nn.Parameter(torch.randn(1)))
        self.register_buffer('running_grad_ratio', None)

        modify_network(self)
        print("22222222222222222")
      


    def find_modules(self):
        return [m for m in self.modules() if isinstance(m, ResBlock)]

    def sparse_param(self, module):
        # embed()
        param1 = module.state_dict(keep_vars=True)['body.0.1.weight']
        param2 = module.state_dict(keep_vars=True)['body.3.1.weight']
        return param1, param2

    def compress(self, **kwargs):
        for module_cur in self.find_modules():
            compress_module_param(module_cur, self.args.remain_percentage, self.args.threshold, self.args.p1_p2_same_ratio)

 

    def set_channels(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.out_channels, m.in_channels = m.weight.size()[:2]



