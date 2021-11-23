import torch
from torch.nn.utils import prune
import torch.nn as nn
import numpy as np

#Pruning crieterion for unstructured threshold pruning
class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold


#Computing sparsity information of the model
def sparsityRate(model,verb=False):
    out=[]
    tot_pruned=0
    tot_struct_pruned=0
    for m in model.find_modules():
        m1, m2 = model.sparse_param(m)
        projection1, projection2 = m1.weight, m2.weight  
        A1 = projection1.squeeze().t()
        A2 = projection2.squeeze().t()
        c1, c2 = model.dense_param(m)

        v=[]
        for i in range(A1.shape[0]):
            el= float((A1[i,:]==0).sum()/A1[i,:].numel())
            v=v+[el]

            tot_pruned+=(A1.shape[1] + c1.kernel_size[0] * c1.kernel_size[1] * c1.in_channels) * el
            if el==1.0:
                tot_struct_pruned += A1.shape[1] + c1.kernel_size[0] * c1.kernel_size[1] * c1.in_channels

        for i in range(A2.shape[0]):
            el= float((A2[i,:]==0).sum()/A2[i,:].numel())
            v=v+[el]

            tot_pruned+=(A2.shape[1] + c2.kernel_size[0] * c2.kernel_size[1] * c2.in_channels) * el
            if el==1.0:
                tot_struct_pruned += A2.shape[1] + c2.kernel_size[0] * c2.kernel_size[1] * c2.in_channels
        


        if verb:
            print("in module ",m,"\n sparsity of  ", v)
        else:
            print("\n sparsity of  ", v)
        out=out+[v]
           

    return out,(tot_pruned,tot_struct_pruned)


#method that prune neurons of the model based on their sparsity
def thresholdNeuronPruning(module,mask,threshold=0.95):

    #Fully Connected layers
   if isinstance(module,nn.Linear):
        for i in range(module.out_features):
            if float((module.weight[i,:]==0).sum()/module.weight[i,:].numel())>threshold:
                mask[i]=0

    #Convolutional layers    
   if isinstance(module,nn.Conv2d):
        for i in range(module.out_channels):
            if float((module.weight[i,:,:,:]==0).sum()/module.weight[i,:,:,:].numel())>threshold:
                mask[i]=0
            
   return 1-mask.sum()/mask.numel()            


# compute the maximum weight for each neuron
def maxVal(model): 
    out=[]
    for m in model.modules():
          #Fully Connected layers
          if isinstance(m,nn.Linear):
                # for i in range(m.out_features):  
                #     v=v+ [float(torch.norm(m.weight[i,:],np.inf))]
                v=torch.norm(m.weight,dim=1,p=np.inf)
          else:
          #Convolutional layers
                if isinstance(m,nn.Conv2d):
                # for i in range(m.out_channels):  
                #     v=v+[float(torch.norm(m.weight[i,:,:,:],np.inf))]
                    v=torch.norm(m.weight,dim=(1,2,3),p=np.inf)
                else:
                    continue

          print("\nmax weight is ", v)
          out+=[v]
    return out


def layerwise_M(model, const = False, scale = 1.0):
       Mdict={}
       if const:
           for m in model.modules():
                if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                    Mdict[m]=1.0
       else:
            for m in model.modules():
                if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                    Mdict[m]=torch.norm(m.weight,p=np.inf).item() * scale

       return Mdict

def noReg(net, loss, lamb=0.1):
    return loss,0

