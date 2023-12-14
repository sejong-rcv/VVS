import tqdm
import torch
import random
import numpy as np
import os
import tarfile
import json
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model.feature_extractor import FeatureExtractor
from torch.utils.data import DataLoader
from datasets.generators import TripletGenerator, ToTensorNormalize, DatasetGenerator
from torchvision import transforms
from arg import arg_func
from datasets import FIVR, CC_WEB_VIDEO


def collate_custom(batch):
    anc = []
    pos = []
    neg = []
    mas = []

    aug = []
    iter_gi = []

    aid = []
    pid = []
    nid = []

    for b in batch:
        if b[0].ndim != 1:
            buffer = []
            for b_ in b:
                if isinstance(b_, np.ndarray):
                    buffer.append(torch.from_numpy(b_.copy()))
                elif isinstance(b_, int) or isinstance(b_, np.int64):
                    buffer.append(torch.tensor(b_))
                else:
                    buffer.append(b_)

            anc.append(buffer[0].unsqueeze(0))
            pos.append(buffer[1].unsqueeze(0))
            neg.append(buffer[2].unsqueeze(0))
            mas.append(buffer[3].unsqueeze(0))


            if len(b) == 9:
                aid.append(buffer[6].unsqueeze(0))
                pid.append(buffer[7].unsqueeze(0))
                nid.append(buffer[8].unsqueeze(0))

            aug.append(buffer[4])
            iter_gi.append(buffer[5])

    if len(anc)==0:
        return [None for i in range(len(b)+1)]
    else: 
        anc = torch.vstack(anc)
        pos = torch.vstack(pos)
        neg = torch.vstack(neg)
        mas = torch.vstack(mas)

        if len(b) == 9:
            aid = torch.vstack(aid)
            pid = torch.vstack(pid)
            nid = torch.vstack(nid)
            return anc, pos, neg, mas, aug, iter_gi, aid, pid, nid
        else: 
            return anc, pos, neg, mas, aug, iter_gi
            

def mkdir(path):
    if os.path.isdir(path) is False:
        os.mkdir(path)

def triplet_extract_func(args, model):
    composed = transforms.Compose([ToTensorNormalize()])
    if args.retrieve_triplet:
        generator = TripletGenerator(transform=composed, return_id=True, fixed=args.extract_fixed)
    else: 
        import pdb; pdb.set_trace()

    mkdir(args.save_path)
    tar = tarfile.open( os.path.join(args.save_path, 'sources.tar'), 'w' )
    tar.add( 'datasets' )
    tar.add( 'model' )
    curr_file = os.listdir(os.getcwd())
    curr_file = [tar.add(i) for i in curr_file if os.path.isdir(i) is False]
    tar.close()
    
    with open(os.path.join(args.save_path,'args.txt'), 'w') as f:
        json.dump(dict(vars(args)), f, indent=2)

    print("[Info] Generating directories")
    feats_path = os.path.join(args.save_path, "features"); mkdir(feats_path)
    mask_path = os.path.join(args.save_path, "mask"); mkdir(mask_path)
    aug_path = os.path.join(args.save_path, "fixed_extraction_aug"); mkdir(aug_path)
    
    pa = generator.video_paths
    stat_dict = {}
    for i in list(pa.keys()):
        val = None
        i_dir = os.path.join(feats_path, str(i))
        if os.path.isdir(i_dir) is True: 
            dir_list = os.listdir(i_dir)
            subil = [int(dli.split(".")[0]) for dli in dir_list]
            if len(subil)!=0:
                val = max(subil)
                print("{}_{}".format(i, val))
        stat_dict.update({i: val})

    log_txt_path = os.path.join(args.save_path, "iteration_log.txt")
    if os.path.isfile(log_txt_path) is False:
        ftxt = open(log_txt_path, "w")
        ftxt.write("iter anc pos neg\n")
        ftxt.close()

    global_start = time.time()
    total_split = []
    mem_cal_times = []
    gi = 0

    model.eval()
    # Main loop
    for e in range(1000):
        # Sample triplets and start triplet generator
        if (args.extract_fixed is not None):
            generator.sample_triplets(args.extract_fixed)
        else:
            generator.sample_triplets(1000)

        loader = DataLoader(generator, batch_size=1, num_workers=8, collate_fn=collate_custom)
        p_bar = tqdm.tqdm(enumerate(loader), desc='gpu:{} epoch:{}'.format(args.gpu, e), unit='iter', total=len(loader))
        for batch_i, data_out in p_bar:
            if args.iterations<=gi:
                break

            A       = data_out[0]
            P       = data_out[1]
            N       = data_out[2]
            mask    = data_out[3]
            aug     = data_out[4]
            iter_gi = data_out[5]
            aid     = data_out[6]
            pid     = data_out[7]
            nid     = data_out[8]

            if (A is not None) and (P is not None) and (N is not None) and (mask is not None) and (aug is not None):
                gi+=1
          
                A = A.cuda().squeeze(0).permute(1,0,2,3)
                P = P.cuda().squeeze(0).permute(1,0,2,3)
                N = N.cuda().squeeze(0).permute(1,0,2,3)
                aug = aug[0]

                with torch.no_grad():
                    afeats = model(A).cpu()
                    pfeats = model(P).cpu()
                    nfeats = model(N).cpu()
                
                ai = aid.item()
                a_key, asi = feature_saver(ai, afeats, feats_path, stat_dict)
                stat_dict[ai] = asi

                pi = pid.item()
                p_key, psi = feature_saver(pi, pfeats, feats_path, stat_dict)
                stat_dict[pi] = psi

                ni = nid.item()
                n_key, nsi = feature_saver(ni, nfeats, feats_path, stat_dict)
                stat_dict[ni] = nsi

                ftxt = open(log_txt_path, "a")
                if (args.extract_fixed is not None):
                    iter_flag = int(iter_gi[0])
                else:
                    iter_flag = gi
                line = "{} {} {} {}\n".format(iter_flag, a_key, p_key, n_key)

                aug.update({"index" : {"anchor": a_key, "positive": p_key, "negative": n_key}})
                curr_aug_path = os.path.join(aug_path, "iter{:07d}.json".format(gi))
                if os.path.isfile(curr_aug_path) is False:
                    with open(curr_aug_path, 'w') as f:
                        json.dump(aug, f, indent=4)
                ftxt.write(line)
                ftxt.close()

                mask.long()

                if (args.extract_fixed is not None):
                    torch.save(mask.long(), os.path.join(mask_path, "iter{:07d}.pt".format(int(iter_gi[0]))))
                else:
                    torch.save(mask.long(), os.path.join(mask_path, "iter{:07d}.pt".format(gi)))

                curr_size = get_dir_size(args.save_path)

                mem_start = time.time()
                mem_tb = (curr_size/(1024**4))
                if mem_tb > 1:
                    print("*"*20)
                    print("\nMemory Alarm!!!\n")
                    print("*"*20)
                    import pdb; pdb.set_trace()
                mem_cal_times.append(time.time()-mem_start)
                
                global_end = time.time()
                duration = global_end - global_start - sum(mem_cal_times)
                per_batch = duration/((e) * len(loader) + (batch_i+1))
                per_iter = duration/gi
                p_bar.set_description( \
                    "gi:{:6d}, {:7.3f}s/batch, {:7.3f}s/iter, {:7.3f}s/mcal, {:8.6f} TB, a:{:6d}, p:{:6d}, n:{:6d}, rnd:{:8.6f}"\
                    .format(gi, per_batch, per_iter, sum(mem_cal_times)/len(mem_cal_times), mem_tb, aid.item(), pid.item(), nid.item(), aug['rnd'])
                )
            elif (args.extract_fixed is not None):
                print("\n")
                print("*"*20)
                print(aid,pid,nid)
                print("\nAll videos must be readable!!!\n")
                print("\nPlease Check Video or Turn off the extract_fixed option\n")
                print("*"*20)
                print("\n")

        if args.iterations<=gi:
            break


def eval_extract_func(args, model):

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

    mkdir(args.save_path)
    tar = tarfile.open( os.path.join(args.save_path, 'sources.tar'), 'w' )
    tar.add( 'datasets' )
    tar.add( 'model' )
    curr_file = os.listdir(os.getcwd())
    curr_file = [tar.add(i) for i in curr_file if os.path.isdir(i) is False]
    tar.close()
    
    with open(os.path.join(args.save_path,'args.txt'), 'w') as f:
        json.dump(dict(vars(args)), f, indent=2)

    print("[Info] Generating directories")
    feats_path = os.path.join(args.save_path, "features"); mkdir(feats_path)

    model.eval()

    composed = transforms.Compose([ToTensorNormalize()])

    generator = DatasetGenerator(dataset = args.dataset, videos=path['query'], 
        transform=composed, load_feats=None)
    loader = DataLoader(generator, num_workers=8, shuffle=False)

    D_t = []
    F_t = []
    S_t = []

    ts = 0
    fs = 0
    total_number = len(loader)

    p_bar = tqdm.tqdm(loader)
    with torch.no_grad():
        start = time.time()
        for video in p_bar:
            d_time = time.time() - start
            start = time.time()

            vid_tensor, vid, load_time = video
            if vid_tensor.dim()==2:
                print("[Invalid Video!] {}".format(vid[0]))
                fs+=1
                continue
            ts+=1

            vid_tensor = vid_tensor.cuda().squeeze(0).permute(1,0,2,3)
            feats = model(vid_tensor).cpu()
            
            f_time = time.time() - start
            start = time.time()
            
            torch.save(feats, os.path.join(feats_path, "{}.pt".format(vid[0])))
            
            s_time = time.time() - start
            start = time.time()

            D_t.append(d_time)
            F_t.append(f_time)
            S_t.append(s_time)

            curr_size = get_dir_size(args.save_path)
            mem_tb = (curr_size/(1024**4))

            descline = "Saved: {}/{} (pass:{}), {:5.3f}s/data, {:5.3f}s/feats, {:5.3f}s/save, {:8.6f}TB".format(
                ts, total_number, fs, 
                np.mean(D_t), np.mean(F_t), np.mean(S_t),
                mem_tb
            )
            p_bar.set_description(descline)


    generator = DatasetGenerator(dataset = args.dataset, videos=path['database'], 
        transform=composed, load_feats=None)
    loader = DataLoader(generator, num_workers=8, shuffle=False)
    
    D_t = []
    F_t = []
    S_t = []

    ts = 0
    fs = 0
    total_number = len(loader)

    p_bar = tqdm.tqdm(loader)
    with torch.no_grad():
        start = time.time()
        for video in p_bar:
            d_time = time.time() - start
            start = time.time()

            vid_tensor, vid, load_time = video
            if vid_tensor.dim()==2:
                print("[Invalid Video!] {}".format(vid[0]))
                fs+=1
                continue
            ts+=1

            vid_tensor = vid_tensor.cuda().squeeze(0).permute(1,0,2,3)
            feats = model(vid_tensor).cpu()

            f_time = time.time() - start
            start = time.time()

            torch.save(feats, os.path.join(feats_path, "{}.pt".format(vid[0])))

            s_time = time.time() - start
            start = time.time()

            D_t.append(d_time)
            F_t.append(f_time)
            S_t.append(s_time)

            curr_size = get_dir_size(args.save_path)
            mem_tb = (curr_size/(1024**4))

            descline = "Saved: {}/{} (pass:{}), {:5.3f}s/data, {:5.3f}s/feats, {:5.3f}s/save, {:8.6f}TB".format(
                ts, total_number, fs, 
                np.mean(D_t), np.mean(F_t), np.mean(S_t),
                mem_tb
            )
            p_bar.set_description(descline)

def extract_func(args):
    model = FeatureExtractor(
        network=args.feature_backbone)
    model = model.cuda()

    if args.dataset in ["vcdb"]:
        triplet_extract_func(args, model)
    else: 
        eval_extract_func(args, model)

def get_dir_size(path='.'):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def feature_saver(cid, feats, feats_path, stat_dict):
    c_folder = os.path.join(feats_path, str(cid))
    mkdir(c_folder)
    c_path = os.path.join(c_folder, "{}.pt")
    c_key = "{}_{}"

    if stat_dict[cid] is None:
        csi = 0
        torch.save(feats, c_path.format(csi))
    else:
        is_exist = False
        csi = stat_dict[cid]+1
        for si in range(csi):
            f_buffer = torch.load(c_path.format(si))
            try:
                is_same = torch.all(feats==f_buffer).item()
            except:
                import pdb; pdb.set_trace()
            if is_same is True:
                is_exist = True
                csi = si
                break
        if is_exist is False:
            torch.save(feats, c_path.format(csi))

    c_key = c_key.format(cid, csi)
    return c_key, csi



if __name__ == '__main__':

    # For reproducibility
    random_seed = 0
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    args = arg_func()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    extract_func(args)
