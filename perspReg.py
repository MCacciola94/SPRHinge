import torch
import numpy as np
from spr_reg import spr_comp

class myTools: 
   def __init__(self,alpha,M):
       self.alpha=alpha
       self.M=M
    #Computation of the current factor
   def myReg(self,net, loss, lamb = 0.1):
        reg = 0 
        tot=0
        for m in net.find_modules():
            m1, m2 = net.sparse_param(m)
            projection1, projection2 = m1.weight, m2.weight          
            A1 = projection1.squeeze().t()
            A2 = projection2.squeeze().t()
            c1, c2 = net.dense_param(m)
            numel1 = A1.shape[1] + c1.kernel_size[0] * c1.kernel_size[1] * c1.in_channels
            numel2 = A2.shape[1] + c2.kernel_size[0] * c2.kernel_size[1] * c2.in_channels
            reg += numel1*spr_comp(A1, self.alpha, self.M[m1])
            reg += numel2*spr_comp(A2, self.alpha, self.M[m2])
            tot += numel1 + numel2 
              
                        
        loss = loss +lamb* reg/tot
        reg=(lamb* reg/tot).item()
        return loss, reg
    
    
    
  
    
    #Computation of y variables gradients and values
   def yGrads(self,model):
        grads=[]
        Y=[]
        for m in model.modules():
          if isinstance(m, torch.nn.Conv2d): 
              g,y=self.yGrad(m)
              grads=grads+[(g)]
              Y=Y+[y]
        return grads, Y
    
   def yGrad(self,m):
        M=self.M[m]
        alpha=self.alpha
        out=[]
        ys=[]

        #Fully Connected layers
        if isinstance(m,torch.nn.Linear):
         
            for i in range(m.out_features):
                a=max(torch.abs(m.weight[i,:]).view(m.weight[i,:].numel()))
                b=torch.norm(m.weight[i,:])*np.sqrt(alpha/(1-alpha))
                y=min(max(a/M,b),torch.tensor(1).cuda())
                ys=ys+[y.item()]
                out=out+[(-alpha*(torch.norm(m.weight[i,:])**2)/y**2+1-alpha).item()]
                
        #Convolutional layers       
        if isinstance(m,torch.nn.Conv2d):
            
            for i in range(m.out_channels):
                a=max(torch.abs(m.weight[i,:,:,:]).view(m.weight[i,:,:,:].numel()))
                b=torch.norm(m.weight[i,:,:,:])*np.sqrt(alpha/(1-alpha))
                y=min(max(a/M,b),torch.tensor(1).cuda())
                ys=ys+[y.item()]
                out=out+[(-alpha*(torch.norm(m.weight[i,:,:,:])**2)/y**2+1-alpha).item()]
        return out,ys
    

#-------------------------------------------------------------------------------------------------

#Code to retrive Y variables information from log files
class yDatas:
  
    def __init__(self,logFile):
        self.y=[]
        self.grads=[]
        self.setupName=logFile
        f=open(logFile,"r")
        l=f.readline()
        while not("alpha" in l):
            l=f.readline()
        s=l.split() 
        self.alpha=s[1]
        
        while not("M" in l):
            l=f.readline()
        s=l.split()
        self.M=s[1]
        while not("Y gradients" in l):
            l=f.readline()
        s=l.split()
        temp=[]
        for el in s[slice(2,len(s))]:
            if "[" in el:
                self.grads.append(temp)
                temp=[]
            el=el.replace("[","")
            el=el.replace("]","")
            el=el.replace(",","")
            temp.append(float(el))
        self.grads.append(temp)
        l=f.readline()
        while not("Y" in l):
            l=f.readline()
        s=l.split()
        temp=[]
        for el in s[slice(2,len(s))]:
            if "[" in el:
                self.y.append(temp)
                temp=[]
            el=el.replace("[","")
            el=el.replace("]","")
            el=el.replace(",","")
            temp.append(float(el))
        self.y.append(temp)
        f.close()