import torch
import numpy as np


def spr_comp(weight, alpha, M):

      const = (torch.sqrt(torch.Tensor([alpha/(1-alpha)]))).cuda()

      #Hinge matrix               
      norminf = torch.norm(weight,dim=1,p=np.inf)
      norm2 = torch.norm(weight,dim=1,p=2)



      bo1 = torch.max(norminf/M,const*norm2)>=1
      reg1 = norm2**2+1-alpha

      bo2 = norminf/M<=const*norm2
      reg2 = const*norm2*(1+alpha)

      eps = (torch.zeros(norminf.size())).cuda()
      eps = eps+1e-10
      reg3 = norm2**2/(torch.max(eps,norminf))*M+(1-alpha)*norminf/M

      bo2 = torch.logical_and(bo2, torch.logical_not(bo1))
      bo3 = torch.logical_and(torch.logical_not(bo2), torch.logical_not(bo1))

      reg = (bo1*reg1+bo2*reg2+bo3*reg3).sum()

      return reg
    
    
    
  
    

    

#-------------------------------------------------------------------------------------------------
