import os
import h5py
import torch
import tqdm
import numpy as np
import pickle

from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize
import torch.nn.functional as F


def load_feats(rmac_file):

    with h5py.File(rmac_file, 'r') as f:
        rmac_features = f['rmac-features'][:]
    rmac_features = np.transpose(rmac_features, (0, 2, 3, 1))
    rmac_features = np.ascontiguousarray(rmac_features)

    return rmac_features

def return_allfiles(dir_path):
    file_pathes = []
    for (root, directories, files) in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_pathes.append(file_path)
    return sorted(file_pathes)

if __name__ == "__main__":
    save_path = "data/vcdb/pca.pkl"
    batch_size = 5000

    root = "features/vcdb_resnet50_l4imac/features/"
    target_list = return_allfiles(root)
    target_list = target_list[:-(len(target_list) % batch_size)]

    transformer = IncrementalPCA(n_components=3840, batch_size=batch_size, whiten=True)

    iteration = len(target_list) // batch_size
    for i in range(iteration):
        curr_list = target_list[i*batch_size: (i+1)*batch_size]
        curr_feats = []
        for j in tqdm.tqdm(curr_list, desc="{} / {}".format(i+1, iteration)):
            rmac_features = torch.load(j)
            split_features = torch.split(rmac_features, [256, 512, 1024, 2048], dim=-1)
            norm_features = torch.cat([F.normalize(i, p=2, dim=-1) for i in split_features], dim=-1)
            features = F.normalize(norm_features, p=2, dim=-1)

            features = torch.mean(features, dim=1)
            features = F.normalize(features, dim=-1)
            features = torch.mean(features, dim=0).unsqueeze(0)
            features = F.normalize(features, dim=-1)
            curr_feats.append(features)
        curr_feats = torch.cat(curr_feats, dim=0)

        transformer.partial_fit(curr_feats.numpy())
   
    f = open(save_path.format(len(target_list)),'wb')
    pickle.dump(transformer,f)
