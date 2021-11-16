from torch.nn.utils import prune
import torch
from importlib import import_module
import aux_tools as at

model_names = ["resnet20_hinge_svd"]

def is_available(name):
    return name in model_names

def load_arch(name, num_classes, resume = ""):
    if not(is_available(name)):
        print("Architecture requested not available")
        return None

    mod = import_module("hinge_resnet_basic_svd")
    class Args:
        pass

    args = Args()
    args.depth = 20
    args.downsample_type= "A"
    args.n_class = num_classes
    args.init_method = "svd2" 
    model = mod.make_model(args)
    model.cuda()





    if not(resume == "")  and already_pruned:
        for m in model.modules(): 
            if hasattr(m, 'weight'):
                pruning_par=[((m,'weight'))]

                if hasattr(m, 'bias') and not(m.bias==None):
                    pruning_par.append((m,'bias'))

                prune.global_unstructured(pruning_par, pruning_method=at.ThresholdPruning, threshold=1e-18)

                
    # optionally resume from a checkpoint

    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint['best_prec1'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    return model