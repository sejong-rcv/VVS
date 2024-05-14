import os
import cv2

import tqdm
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

class ToTensorNormalize(object):
    def __init__(self, use_ms=True):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.use_ms = use_ms

    def __call__(self, frames):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(3, 0, 1, 2)
        frames /= 255

        if self.use_ms:
            for fi, (f, m, s) in enumerate(zip(frames, self.mean, self.std)):
                frames[fi] = (f - m) / s

        return frames

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

class DatasetGenerator(Dataset):
    def __init__(self, vid_dir, transform=None, fps=1, cc_size=224, rs_size=256):
        super(DatasetGenerator, self).__init__()

        self.vid_dir = vid_dir
        self.videos = os.listdir(self.vid_dir)
        self.transform = transform

        self.fps = fps
        self.cc_size = cc_size
        self.rs_size = rs_size

    def __len__(self):
        return len(self.videos)

    def center_crop(self, frame, desired_size):
        if frame.ndim == 1:
            return frame
        elif frame.ndim == 3:
            old_size = frame.shape[:2]
            top = int(np.maximum(0, (old_size[0] - desired_size)/2))
            left = int(np.maximum(0, (old_size[1] - desired_size)/2))
            return frame[top: top+desired_size, left: left+desired_size, :]
        else: 
            old_size = frame.shape[1:3]
            top = int(np.maximum(0, (old_size[0] - desired_size)/2))
            left = int(np.maximum(0, (old_size[1] - desired_size)/2))
            return frame[:, top: top+desired_size, left: left+desired_size, :]


    def resize_frame(self, frame, desired_size):
        min_size = np.min(frame.shape[:2])
        ratio = desired_size / min_size
        frame = cv2.resize(frame, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        return frame

    def load_video(self, video, slow_mo=1, all_frames=False, fps=1, cc_size=224, rs_size=256): 
        cv2.setNumThreads(1) 
        cap = cv2.VideoCapture(video)
        fps_div = fps
        fps = cap.get(cv2.CAP_PROP_FPS) / slow_mo
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps > 144 or fps is None:
            fps = 25
        frames = []
        for fi in range(frame_count):
            ret = cap.grab()
            if int(fi % round(fps / fps_div)) == 0 or all_frames:
                ret, frame = cap.retrieve()
                if isinstance(frame, np.ndarray):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if rs_size is not None:
                        frame = self.resize_frame(frame, rs_size)
                    frames.append(frame)
                else:
                    break
        cap.release()
        frames = np.array(frames)
        if cc_size is not None:
            frames = self.center_crop(frames, cc_size)
        return frames

    def __getitem__(self, idx):
        video = self.load_video(os.path.join(self.vid_dir, self.videos[idx]), fps=self.fps, cc_size=self.cc_size, rs_size=self.rs_size)
        raw_frames = video.copy()
        vid = self.videos[idx]
        try:
            if video is None:
                return torch.from_numpy(np.array([])), vid, torch.from_numpy(np.array([]))
            else:
                if self.transform:
                    video = self.transform(video)
                return video, vid, raw_frames, 
        except:
            return torch.from_numpy(np.array([])), vid, torch.from_numpy(np.array([]))
            

if __name__ == '__main__':

    root_dir = '' # check the path to root video directory
    save_dir = './easy_distractor'
    if os.path.isdir(save_dir)==False:
        os.mkdir(save_dir)

    mag_thresh = 40 

    backbone_extractor = FeatureExtractor(network='resnet50')
    backbone_extractor = backbone_extractor.cuda()
    backbone_extractor.eval()

    composed = transforms.Compose([ToTensorNormalize()])
    
    generator = DatasetGenerator(root_dir, transform=composed)
    loader = DataLoader(generator, num_workers=0, shuffle=False)

    total_number = len(loader)
    p_bar = tqdm.tqdm(loader)
    with torch.no_grad():
        for video in p_bar:
            vid_tensor, vid, frames = video

            if vid_tensor.dim()==2: # error
                continue

            # extract l4-imac
            vid_tensor = vid_tensor.cuda().squeeze(0).permute(1,0,2,3)
            feat = backbone_extractor(vid_tensor)

            # magnitude 
            feat = feat.mean(dim=1)
            feat_magnitude = torch.norm(feat,dim=-1)

            # magnitude thresholding
            easy_distractors_idx = torch.where(feat_magnitude<mag_thresh)[0]
            sample_idx = easy_distractors_idx.detach().cpu()
            easy_distractor_frames = frames[:,sample_idx,:,:,:].detach().squeeze(0).cpu().numpy()

            # save_to_png
            if os.path.isdir(os.path.join(save_dir,vid[0].split('.')[0]))==False:
                os.mkdir(os.path.join(save_dir,vid[0].split('.')[0]))

            for idx, frame in enumerate(easy_distractor_frames):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(save_dir, vid[0].split('.')[0], f'{idx}.png'), frame)
