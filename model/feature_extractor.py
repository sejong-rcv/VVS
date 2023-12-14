import copy
import time
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FeatureExtractor(nn.Module):
    def __init__(self, network='resnet50', bn_freeze=True, kernel=[28, 14, 6, 3]):
        super(FeatureExtractor, self).__init__()

        if network=='resnet50':
            print("[Option] Backbone : Resnet50")
            self.cnn = models.resnet50(pretrained=True)
        else:
            print("[Error] Wrong Backbone!")
            import pdb; pdb.set_trace()
        
        self.stride = True if sorted(kernel) == sorted([28,14,6,3]) else False
        print("[Option] Stride : ", self.stride)
        self.layers = {'layer1': kernel[0], 'layer2': kernel[1], 'layer3': kernel[2], 'layer4': kernel[3]}
        print("[Option] Layer : ", self.layers)
        if bn_freeze:
            self.cnn.eval()
            print("[Option] Backbone(+batchnorm) is Freezed!")

    def extract_region_vectors(self, x):
        tensors = torch.tensor([]).cuda()
        for nm, module in self.cnn._modules.items():
            if nm not in {'avgpool', 'fc', 'classifier'}:
                x = module(x).contiguous()
                if nm in self.layers:
                    s = self.layers[nm]
                    region_vectors = F.max_pool2d(x, [s,s], int(np.ceil(s/2))) 
                    tensors = torch.cat((tensors, region_vectors), dim=1)
        x = tensors
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        return x
    
    def forward(self, x):
        x = self.extract_region_vectors(x)
        return x
