import tqdm
import torch
import random
import numpy as np
import os
import json
import copy
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model.model import SuppressionWeightGenerationStage, EasyDistractorEliminationStage
from model.feature_extractor import FeatureExtractor
from torch.utils.data import DataLoader
from datasets.generators import ToTensorNormalize, DatasetGenerator
from torchvision import transforms
from arg import arg_func

from sklearn.metrics.pairwise import cosine_similarity
from datasets import FIVR, CC_WEB_VIDEO
from model.model_utils import PCA


def collate_custom(batch):
    anc = []
    pos = []
    neg = []
    mas = []
    for b in batch:
        if b[0].ndim != 1:
            anc.append(b[0].unsqueeze(0))
            pos.append(b[1].unsqueeze(0))
            neg.append(b[2].unsqueeze(0))
            mas.append(torch.from_numpy(b[3].copy()).unsqueeze(0))

    if len(anc)==0:
        return None, None, None, None
    else: 
        anc = torch.vstack(anc)
        pos = torch.vstack(pos)
        neg = torch.vstack(neg)
        mas = torch.vstack(mas)
     
        return anc, pos, neg, mas

def mkdir(path):
    if os.path.isdir(path) is False:
        os.mkdir(path)

def global_logger(global_log, dict_out, header=None):
    for k, v in dict_out.items():
        if v.ndim!=0:
            continue
        val = v.item()

        key = header + "_" + k if header is not None else k

        if key not in global_log:
            global_log.update({key : [val, 1]})
        else:
            scalar, trial = global_log[key]
            new_trial = trial + 1
            new_scalar = (scalar*trial + val) / new_trial
            global_log[key] = [new_scalar, new_trial]
    return global_log
        
def eval_func(args):

    if args.dataset == 'fivr5k':
        dataset = FIVR(version='5k')
        with open('data/fivr/fivr5k_vid.json','r') as f:
            path = json.load(f)
                
    elif args.dataset == 'fivr200k':
        dataset = FIVR(version='200k')
        with open('data/fivr/fivr200k_vid.json','r') as f:
            path = json.load(f)
            
    elif args.dataset == 'cc_web':
        dataset = CC_WEB_VIDEO()
        with open('data/cc_web/cc_web_vid.json','r') as f:
            path = json.load(f)

    if args.feats_load_dir is None:
        backbone_extractor = FeatureExtractor(
            network=args.feature_backbone)
        backbone_extractor = backbone_extractor.cuda()
    
    # Model definition
    tsm_tgm_model = SuppressionWeightGenerationStage(args)
    ddm_model = EasyDistractorEliminationStage(args)
    composed = transforms.Compose([ToTensorNormalize()])

    # Suppression Weight Generation Stage
    if args.load_path is not None: 
        curr = copy.deepcopy(tsm_tgm_model.state_dict())
        loaded = torch.load(args.load_path)
        missing, unexpected = tsm_tgm_model.load_state_dict(loaded["tsm_tgm_model"], strict=False)
        after = tsm_tgm_model.state_dict()

        print("[Loaded]: {}".format(args.load_path))
        for k in curr.keys():
            curr_val = curr[k] 
            after_val = after[k]
            if curr_val.ndim==0:
                if isinstance(curr_val.item(), bool):
                    curr_val = curr_val.long()
            if after_val.ndim==0:
                if isinstance(after_val.item(), bool):
                    after_val = after_val.long()

            if k in missing:
                print("\t[Missing]: {}".format(k))
            elif k in unexpected:
                print("\t[Unexpected]: {}".format(k))
            elif torch.sum(curr_val-after_val).item()!=0:
                print("\t[Loaded]: {}".format(k))
            else:
                print("\t[Not Loaded]: {}".format(k))

    # Easy Distractor Elimination Stage
    if args.load_path is not None: 
        curr = copy.deepcopy(ddm_model.state_dict())
        loaded = torch.load(args.load_path)
        missing, unexpected = ddm_model.load_state_dict(loaded["ddm_model"], strict=False)
        after = ddm_model.state_dict()

        print("[Loaded]: {}".format(args.load_path))
        for k in curr.keys():
            curr_val = curr[k] 
            after_val = after[k]
            if curr_val.ndim==0:
                if isinstance(curr_val.item(), bool):
                    curr_val = curr_val.long()
            if after_val.ndim==0:
                if isinstance(after_val.item(), bool):
                    after_val = after_val.long()

            if k in missing:
                print("\t[Missing]: {}".format(k))
            elif k in unexpected:
                print("\t[Unexpected]: {}".format(k))
            elif torch.sum(curr_val-after_val).item()!=0:
                print("\t[Loaded]: {}".format(k))
            else:
                print("\t[Not Loaded]: {}".format(k))

    tsm_tgm_model = tsm_tgm_model.cuda()
    ddm_model = ddm_model.cuda()

    tsm_tgm_model.eval()
    ddm_model.eval()

    data_out = {
        "qr": {"id": [], "feats":[]}, 
        "db": {"id": []},
        }
    
    generator = DatasetGenerator(dataset = args.dataset, videos=path['query'], 
        transform=composed, load_feats=args.feats_load_dir)
    loader = DataLoader(generator, num_workers=0, shuffle=False)

    total_number = len(loader)
    p_bar = tqdm.tqdm(loader)
    with torch.no_grad():
        start = time.time()
        for video in p_bar:
            vid_tensor, vid, load_time = video

            if vid_tensor.dim()==2:
                continue

            if args.feats_load_dir is None: ## extract feature from raw video

                vid_tensor = vid_tensor.cuda().squeeze(0).permute(1,0,2,3)
                vid_tensor = backbone_extractor(vid_tensor)
                vid_tensor = vid_tensor.unsqueeze(0)

            elif vid_tensor.dim()==3 and vid_tensor.shape[0]==1:
                vid_tensor = vid_tensor.unsqueeze(0)
            
            in_data = {
                "anchor"  : vid_tensor.cuda(),
            }

            if args.vvs_ddm:
                ddm_out = ddm_model(in_data['anchor'])
                if (torch.sigmoid(ddm_out['confidence'])>0.5).float().sum() != 0:
                    uneliminated_index = (torch.sigmoid(ddm_out['confidence'])>0.5).bool()
                    in_data['anchor'] = in_data['anchor'][:,uneliminated_index]
  
            while in_data['anchor'].shape[1] < 4:
                in_data['anchor'] = torch.cat([in_data['anchor'], in_data['anchor']], 1)
                
            feats_out = tsm_tgm_model(in_data, is_anc_processed=False)
            data_out["qr"]["feats"].append(feats_out["features"])
            data_out["qr"]["id"].append(vid[0][0] if len(vid)!=1 else vid[0])

    data_out.update({"sim_v": dict({query: dict() for query in data_out["qr"]["id"]})})

    generator = DatasetGenerator(dataset = args.dataset, videos=path['database'], 
        transform=composed, load_feats=args.feats_load_dir)
    loader = DataLoader(generator, num_workers=0, shuffle=False)

    global_log = {}
    p_bar = tqdm.tqdm(loader)
    with torch.no_grad():
        start = time.time()
        for video in p_bar:
            d_time = time.time() - start
            start = time.time()

            vid_tensor, vid, load_time = video
           
            if vid_tensor.dim()==2:
                continue
            
            if args.feats_load_dir is None: ## extract feature from raw video
                vid_tensor = vid_tensor.cuda().squeeze(0).permute(1,0,2,3)
                vid_tensor = backbone_extractor(vid_tensor)
                vid_tensor = vid_tensor.unsqueeze(0)
            
            elif vid_tensor.dim()==3 and vid_tensor.shape[0]==1:
                vid_tensor = vid_tensor.unsqueeze(0)
            
            in_data = {
                "anchor"  : vid_tensor.cuda(),
            }

            if args.vvs_ddm:
                ddm_out = ddm_model(in_data['anchor'])
                if (torch.sigmoid(ddm_out['confidence'])>0.5).float().sum() != 0:
                    uneliminated_index = (torch.sigmoid(ddm_out['confidence'])>0.5).bool()
                    in_data['anchor'] = in_data['anchor'][:,uneliminated_index]

            while in_data['anchor'].shape[1] < 4:
                in_data['anchor'] = torch.cat([in_data['anchor'], in_data['anchor']], 1)

            feats_out = tsm_tgm_model(in_data, is_anc_processed=False)
            data_out["db"]["id"].append(vid[0][0] if len(vid)!=1 else vid[0])

            f_time = time.time() - start
            start = time.time()

            for qi, qfeats in enumerate(data_out["qr"]["feats"]):
                qid = data_out["qr"]["id"][qi]
                sim_v = tsm_tgm_model.calculate_pair_sim(qfeats, feats_out["features"])
                data_out["sim_v"][qid][vid[0][0] if len(vid)!=1 else vid[0]] = sim_v.item()

            s_time = time.time() - start
            start = time.time()

            time_out = {
                "Time_data"  : torch.tensor(d_time),
                "Time_model" : torch.tensor(f_time),
                "Time_sim" : torch.tensor(s_time),
            }

            global_log = global_logger(global_log, time_out) 
            logline = ""
            for k, v in global_log.items():
                spt = k.split("_")
                spt = "_".join([spt[0][0], spt[1][0:2]])
                spt += ":{:5.3f}, ".format(v[0])
                logline += spt
            logline = logline[: -2]
            p_bar.set_description(logline)
            
    if args.load_path is not None:
        center_name = args.load_path.split("/")[-1].split(".")[0]
    else: 
        center_name = "m00000000"
    mkdir(os.path.join(args.save_path, "eval", center_name))

    all_db = []
    all_db.extend(data_out["qr"]["id"])
    all_db.extend(data_out["db"]["id"])
    all_db = set(all_db)    
    
    for k, v in data_out.items():
        if "sim" not in k:
            continue
        txt_name = k
        if args.dataset == 'fivr200k':
            txt_name = txt_name + '_200k'
        save_name = os.path.join(args.save_path, "eval", center_name, "{}.txt".format(txt_name))
        eval_logger(v, all_db, save_name, dataset, None, long_term=args.long_term_retrieval)

def eval_logger(sim, all_db, save_name, dataset, num_dict=None, long_term=False):
    print("[Save] -> {}".format(save_name))
    if args.dataset == 'fivr5k' or args.dataset == 'fivr200k':
        mAP, mAP_log = dataset.evaluate(sim, all_db, num_dict=num_dict)
        np.savetxt(save_name, mAP_log, fmt="%s")
    elif args.dataset == 'cc_web':
        mAP = dataset.evaluate(sim, all_db)
        save_name = save_name.replace('txt','json')
        json.dump(mAP, open(save_name, 'w', encoding='utf-8'), indent="\t")

if __name__ == '__main__':
    random_seed = 0
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    args = arg_func()
    
    torch.use_deterministic_algorithms(False)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
    
    eval_func(args)
    
