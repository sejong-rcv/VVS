import math
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops


class VideoNormalizer(nn.Module):
    def __init__(self):
        super(VideoNormalizer, self).__init__()
        self.scale = nn.Parameter(torch.Tensor([255.]), requires_grad=False)
        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]), requires_grad=False)

    def forward(self, video):
        video = ((video.permute(1,0,2,3).float() / self.scale) - self.mean) / self.std

        return video.permute(0, 3, 1, 2)


class TensorDot(nn.Module):
    def __init__(self, pattern='iak,jbk->iabj', metric='cosine'):
        super(TensorDot, self).__init__()
        self.pattern = pattern
        self.metric = metric

    def forward(self, query, target):
        if self.metric == 'cosine':
            sim = torch.einsum(self.pattern, [query, target])
        elif self.metric == 'euclidean':
            sim = 1 - 2 * torch.einsum(self.pattern, [query, target])
        elif self.metric == 'hamming':
            sim = torch.einsum(self.pattern, query, target) / target.shape[-1]

        return sim


class PCA(nn.Module):
    def __init__(self, file="data/vcdb/pca.pkl", n_components=3840, whitening=True):
        super(PCA, self).__init__()
        f = open(file,'rb')
        pca_model = pickle.load(f)

        self.whitening = whitening
        self.mean = nn.Parameter(torch.from_numpy(pca_model.mean_), requires_grad=False)
        self.components = nn.Parameter(torch.from_numpy(pca_model.components_[: n_components].T), requires_grad=False)
        self.denominator = nn.Parameter(torch.from_numpy(np.sqrt(pca_model.explained_variance_[: n_components])), requires_grad=False)

        
    def forward(self, logits):
        out = torch.matmul(logits - self.mean, self.components) 
        if self.whitening:
            out = out / self.denominator
        out = out.float()
        out = F.normalize(out, p=2, dim=-1)

        return out


class ChamferSimilarity(nn.Module):
    def __init__(self, symmetric=False, axes=[1, 0], sim_seq="max_mean"):
        super(ChamferSimilarity, self).__init__()
        self.sim_seq = sim_seq
        if symmetric:
            self.sim_fun = lambda x: self.symmetric_chamfer_similarity(x, axes=axes)
        else:
            self.sim_fun = lambda x: self.chamfer_similarity(x, max_axis=axes[0], mean_axis=axes[1])

    def chamfer_similarity(self, s, max_axis=1, mean_axis=0):
        if self.sim_seq=="max_mean":
            s = torch.max(s, max_axis, keepdim=True)[0]
        elif self.sim_seq=="mean_mean":
            s = torch.mean(s, max_axis, keepdim=True)
        return torch.mean(s, mean_axis, keepdim=True).squeeze(max_axis).squeeze(mean_axis)
        
    def symmetric_chamfer_similarity(self, s, axes=[0, 1]):
        return (self.chamfer_similarity(s, max_axis=axes[0], mean_axis=axes[1]) +
                self.chamfer_similarity(s, max_axis=axes[1], mean_axis=axes[0])) / 2
    
    def forward(self, s):
        return self.sim_fun(s)


class VideoComperator(nn.Module):
    def __init__(self, use_dcn=False):
        super(VideoComperator, self).__init__()

        conv = nn.Conv2d if use_dcn==False else DeformableConv2d

        self.rpad1 = nn.ConstantPad2d(1,0)
        self.conv1 = conv(1, 32, 3)
        self.pool1 = nn.MaxPool2d((2, 2), 2)

        self.rpad2 = nn.ConstantPad2d(1,0)
        self.conv2 = conv(32, 64, 3)
        self.pool2 = nn.MaxPool2d((2, 2), 2)

        self.rpad3 = nn.ConstantPad2d(1,0)
        self.conv3 = conv(64, 128, 3)

        self.fconv = conv(128, 1, 1)
        if use_dcn==False:
            self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, sim_matrix):
        sim = self.rpad1(sim_matrix)
        sim = F.relu(self.conv1(sim))
        sim = self.pool1(sim)

        sim = self.rpad2(sim)
        sim = F.relu(self.conv2(sim))
        sim = self.pool2(sim)

        sim = self.rpad3(sim)
        sim = F.relu(self.conv3(sim))
        sim = self.fconv(sim)
        return sim


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, padding_mode='replicate')

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
