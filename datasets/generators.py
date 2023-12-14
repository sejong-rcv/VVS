import os
import glob
import utils
import json
import torch
import random
import numpy as np
import pickle as pk
from torch.utils.data import Dataset
from .augmentation import Augmentor
import time
from PIL import Image
from tqdm import tqdm
import h5py


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


class DatasetGenerator(Dataset):
    def __init__(self, 
        dataset, videos, transform=None, 
        fps=1, cc_size=224, rs_size=256, 
        load_feats=None, idx_only=False, fg=None):
        super(DatasetGenerator, self).__init__()
        self.dataset = dataset
        if 'fivr' in self.dataset:
            self.dataset = 'fivr'
        self.videos = videos
        self.fps = fps
        self.cc_size = cc_size
        self.rs_size = rs_size
        self.transform = transform
        self.load_feats = load_feats
        self.idx_only = idx_only
        self.fg = fg 

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.load_feats is not None: # pre-extraction mode
            vid = self.videos[idx].split('.')[0]
            pt_path = os.path.join(self.load_feats,'features',"{}.pt".format(vid))
            if os.path.isfile(pt_path) is True:
                start = time.time()
                try:
                    feats = torch.load(pt_path).squeeze(0)
                except:
                    feats = torch.from_numpy(np.array([]))
                    return feats, vid, 0
                end = time.time()
                if self.fg is not None:
                    if vid in self.fg:
                        vid = (vid, self.fg[vid])
                    else: 
                        vid = (vid, "")
                return feats, vid, (end-start)
            else: 
                feats = torch.from_numpy(np.array([]))
                return feats, vid, 0
        else: # oneline-extraction mode
            try:
                vid = self.videos[idx]
                path = os.path.join('videos',self.dataset,self.videos[idx])
                start = time.time()
                video = utils.load_video(path, fps=self.fps, cc_size=self.cc_size, rs_size=self.rs_size)
                end = time.time()
                if video is None:
                    return torch.from_numpy(np.array([])), vid, 0
                else:
                    if self.transform:
                        video = self.transform(video)
                    return video, vid, (end - start)
            except:
                return torch.from_numpy(np.array([])), vid, 0


class TripletGenerator(Dataset):
    def __init__(self, 
        transform, dataset="vcdb", 
        data_root="/raid/datasets/vcdb", window=64, 
        return_id=True, fixed=None):

        super(TripletGenerator, self).__init__()
        if dataset != "vcdb":
            print("Only support vcdb for generating triplet!")
            import pdb; pdb.set_trace()
        
        # The triplet pools are loaded from a pickle file that contains a dictionary with the following key-value pairs:
        # 1. 'pool1' - a list whose entries are dictionaries that contains the positive pairs and their hard negatives
        # 2. 'pool2' - a list whose entries are dictionaries that contains single videos and their hard negatives
        # 3. 'index' - a dictionary that maps the indexes to the VCDB video ids.
        self.triplets = pk.load(open('data/vcdb/triplets.pk', 'rb'), encoding='latin1')
        self.indices = []
        self.return_id = return_id

        # Dictionary that contains the video indexes as keys and the paths to videos as values
        # e.g. video_paths[123] = '/path/to/videos/123.mp4'
        with open('data/vcdb/pathes.pickle', 'rb') as f:
            video_paths = pk.load(f)
        video_paths = {k : os.path.join(data_root, v) for k, v in video_paths.items()}

        # Dictionary that contains as keys the video indexes of the video pairs joined with an underscore, and as values
        # the paths to masks. The masks are binary matrices with sizes equal to the corresponding video lengths and
        # contains ones on the video segments that are duplicates and zeros otherwise.
        # e.g. video_paths['0_1'] = '/path/to/masks/0_1.npy'
        masks = dict()
        for triplet in self.triplets['pool1']:
            m = '_'.join(map(str, sorted(triplet['positive_pair'])))
            masks[m] = 'data/vcdb/masks/{}.npy'.format(m)
            if not os.path.isfile(masks[m]):
                print("[Error] Failed to load mask : ", masks[m])
    
        self.augmentor = Augmentor(video_paths, masks, window)
        self.video_paths = video_paths

        self.transform = transform
        self.fixed = fixed is not None
        if self.fixed:
            self.fixed_aug_root = os.path.join("/".join(fixed.split("/")[:-1]), "fixed_extraction_aug")


    def sample_triplets(self, basis):
        # Sample a number of triplets from each triplet pool (Section 4.5 in paper)
        self.indices = []

        if self.fixed:
            self.log_txt = np.loadtxt(basis, dtype=str, delimiter="\n")
            self.log_txt = self.log_txt[1:]

            for pair in tqdm(self.log_txt):
                tmp = pair.split(' ')
                gi = int(tmp[0])
                a_id = int(tmp[1].split('_')[0])
                p_id = int(tmp[2].split('_')[0])
                n_id = int(tmp[3].split('_')[0])
                self.indices.append([gi, a_id, p_id, n_id])
        else: 
            # Draw samples from pool1
            for pair in np.random.choice(self.triplets['pool1'], size=basis, replace=basis > len(self.triplets['pool1'])):
                f = np.random.choice([False, True])
                anchor_id = pair['positive_pair'][int(f)]
                positive_id = pair['positive_pair'][int(not f)]
                negative_id = np.random.choice(pair['hard_negatives'])
                self.indices.append([0, anchor_id, positive_id, negative_id])
            
            # Draw samples from pool2
            for pair in np.random.choice(self.triplets['pool2'], size=basis, replace=basis > len(self.triplets['pool2'])):
                anchor_id = pair['video']
                negative_id = np.random.choice(pair['hard_negatives'])
                self.indices.append([1, anchor_id, anchor_id, negative_id])
            random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        iter_gi, anchor_id, positive_id, negative_id = self.indices[index]
        if self.fixed:
            with open(os.path.join(self.fixed_aug_root, "iter{:07d}.json".format(iter_gi)), 'rb') as f:
                aug = json.load(f)
    
            a, p, n, m, aug = self.augmentor.prepare_triplet(anchor_id, positive_id, negative_id, anchor_id == positive_id, aug)
        else: 
            a, p, n, m, aug = self.augmentor.prepare_triplet(anchor_id, positive_id, negative_id, anchor_id == positive_id)

        if self.transform and (a.ndim!=1):
            a = self.transform(a.copy())
            p = self.transform(p.copy())
            n = self.transform(n.copy())

        if self.return_id:
            return a, p, n, m, aug, iter_gi, anchor_id, positive_id, negative_id
        else:
            return a, p, n, m, aug, iter_gi

class TripletFeatureGenerator(Dataset):
    def __init__(self,
            root_dir="data1",
            log_path="iteration_log.txt", 
            feat_path="features",
            mask_path="mask",
            neg_len=32,
            mag_opt=5):
        super(TripletFeatureGenerator, self).__init__()
        
        start = time.time()
        self.log_txt = np.loadtxt(os.path.join(root_dir, log_path), dtype=str, delimiter="\n")
        print("\t[Log] {} -> Loading time:{:6.2f}s".format(log_path, time.time()-start))
        self.feat_path = os.path.join(root_dir, feat_path)
        self.mask_path = os.path.join(root_dir, mask_path)

        self.log_txt = self.log_txt[1:]
        
        self.neg_len = neg_len
        self.mag_opt = mag_opt # Set the threshold of easy distractor set

        # Set distractor_sampling_ratio
        self.mag_list = [
            '20_lower',
            '20_to_25', 
            '25_to_30', 
            '30_to_35', 
            '35_to_40', 
            '40_to_45', 
            '45_to_50', 
            '50_to_55', 
            '55_to_60', 
            '60_to_65' 
        ]
        self.vcdb_easy_distractor_set = list()
        self.distractor_path = 'features/vcdb_resnet50_l4imac/features'

        # Load distractor_sampling_ratio
        for i in range(self.mag_opt):
            tmp = np.loadtxt('data/vcdb/background_sampling_frames/'+self.mag_list[i]+'.txt', delimiter="\n", dtype='str')
            self.vcdb_easy_distractor_set.extend(list(tmp))
        
        self.vcdb_easy_distractor_set_dict = { i.split('_')[0] : [] for i in self.vcdb_easy_distractor_set }
        for i in self.vcdb_easy_distractor_set:
            vid_id = i.split('_')[0]
            frame_id = '_'.join(i.split('_')[-2:]).replace('.pt', '')
            self.vcdb_easy_distractor_set_dict[vid_id].append(frame_id)

    def __len__(self):
        return len(self.log_txt)

    def __getitem__(self, index):
        
        line = self.log_txt[index]
        line = line.split(" ")
        gi = int(line[0])

        avi, svi = line[1].split("_")
        a_feats = torch.load(os.path.join(self.feat_path, avi, "{}.pt".format(svi)))

        pvi, svi = line[2].split("_")
        p_feats = torch.load(os.path.join(self.feat_path, pvi, "{}.pt".format(svi)))
        
        nvi, svi = line[3].split("_")
        n_feats = torch.load(os.path.join(self.feat_path, nvi, "{}.pt".format(svi)))
    
        if avi != pvi:
            negative_metafree_video_id = np.random.choice(list(self.vcdb_easy_distractor_set_dict.keys()), 1)[0]
            negative_metafree_video_frame = np.random.choice(self.vcdb_easy_distractor_set_dict[negative_metafree_video_id], 1)[0]

            ed_feats = torch.load(os.path.join(self.distractor_path, negative_metafree_video_id, negative_metafree_video_frame.split('_')[0]+'.pt'))
            ed_feats = ed_feats[int(negative_metafree_video_frame.split('_')[1])].unsqueeze(0)
            ed_feats = ed_feats.repeat(self.neg_len,1,1).unsqueeze(0)
        else:
            ed_feats = torch.Tensor([])
        return gi, a_feats, p_feats, n_feats, ed_feats
