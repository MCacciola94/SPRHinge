import torch
import torch.nn as nn
from torch.nn.utils import prune

class EmptyLayer(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, pad, stride):
        super(EmptyLayer, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k_size = k_size
        self.pad = pad
        self.stride = stride
    def forward(self, x):
        b_size, in_ch, height, width = x.shape
        h_new = (height-self.k_size[0]+2*self.pad[0])/self.stride[0]+1
        w_new = (width-self.k_size[1]+2*self.pad[1])/self.stride[1]+1

        return torch.zeros(b_size, self.out_ch, int(h_new), int(w_new)).cuda()

def get_nonzero_index(x):
    n = torch.sum(x, dim=1)
    non_zero_idx = (n!=0).nonzero().squeeze(dim = 1)
    return non_zero_idx



# ========
# Compress module parameters
# ========
def compress_module_param(module):
    body = module._modules['body']
    conv11 = body._modules['0']._modules['0']
    conv12 = body._modules['0']._modules['1']
    conv21 = body._modules['3']._modules['0']
    conv22 = body._modules['3']._modules['1']

    ws1 = conv11.weight.shape
    weight1 = conv11.weight.data.view(ws1[0], -1).t()
    projection1 = conv12.weight.data.squeeze().t()
    projection1_mask = conv12.weight_mask.squeeze().t()

    ws2 = conv21.weight.shape
    weight2 = conv21.weight.data.view(ws2[0], -1).t()
    projection2 = conv22.weight.data.squeeze().t()
    projection2_mask = conv22.weight_mask.squeeze().t()

    pindex1 = get_nonzero_index(projection1_mask)
    
    pindex2 = get_nonzero_index(projection2_mask)
    # print(pindex1.shape)
    #print(pindex1)
    # print(pindex2.shape)
    #print(pindex2)
    if pindex1.shape[0] == 0 :
        print("here")
        body._modules['0'] = EmptyLayer(conv11.in_channels, conv11.out_channels, conv11.kernel_size, conv11.padding, conv11.stride)
        pindex1 = torch.Tensor([i for i in range(projection1_mask.shape[0])]).type(torch.int64).cuda()
        
        #pindex1 = torch.Tensor([0]).type(torch.int64).cuda()
        
    if pindex2.shape[0] == 0:
        #pindex2 = torch.Tensor([0]).type(torch.int64).cuda()
        body._modules['3'] = EmptyLayer(conv21.in_channels, conv21.out_channels, conv21.kernel_size, conv21.padding, conv21.stride)
        pindex2 = torch.Tensor([i for i in range(projection2_mask.shape[0])]).type(torch.int64).cuda()
    prune.remove(conv12, 'weight')
    prune.remove(conv22, 'weight')

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
    #print(body)




def compress(net):
        for module_cur in net.find_modules():
            compress_module_param(module_cur)
        