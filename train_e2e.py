import os
import logging
import copy
import json
import time
import tqdm
import torch
import random
import tarfile
import numpy as np

# from tensorboardX import SummaryWriter
from model.model import SuppressionWeightGenerationStage, EasyDistractorEliminationStage
from torch.utils.data import DataLoader
from datasets.generators import TripletFeatureGenerator
from torchvision import transforms
from arg import arg_func


def collate_custom(batch):

    anc = []
    pos = []
    neg = []
    mas = []
    for b in batch:
        if b[0].ndim != 1:
            buffer = []
            for b_ in b:
                if isinstance(b_, np.ndarray):
                    buffer.append(torch.from_numpy(b_.copy()))
                else: 
                    buffer.append(b_)

            anc.append(buffer[0].unsqueeze(0))
            pos.append(buffer[1].unsqueeze(0))
            neg.append(buffer[2].unsqueeze(0))
            mas.append(buffer[3].unsqueeze(0))
   
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

def train_writer(summary_writer, dict_out, step, header, middle=None):
    for k, v in dict_out.items():
        if isinstance(v, dict):
            continue
        
        if v is None:
            continue
        if v.ndim!=0:
            continue
        val = v.item()
        if val is not None: 
            mid = "" if middle is None else middle+"_"
            summary_writer.add_scalar('{}/{}{}'.format(header, mid, k), val, step)

def global_logger(global_log, dict_out, header=None):
    for k, v in dict_out.items():
        if isinstance(v, dict):
            continue
        if v is None:
            continue

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

def train_func(args):

    if args.feats_load_dir is None:
        print("Train process: only support pre-extraction mode!")
        import pdb; pdb.set_trace()

    ## Model definition by stage due to ease of code implementation (however, learning works at the same time)

    # Suppression Weight Generation Stage Definition
    tsm_tgm_model = SuppressionWeightGenerationStage(args)
    tsm_tgm_model = tsm_tgm_model.cuda()

    # Easy Distractor Elimination Stage Definition
    ddm_model = EasyDistractorEliminationStage(args)
    ddm_model = ddm_model.cuda()


    # Loader definition
    train_generator = TripletFeatureGenerator(root_dir=args.feats_load_dir, log_path="iteration_log.txt", neg_len=args.neg_len, mag_opt=args.mag_opt)

    # Save folder generation
    save_path = args.save_path
    log_path = os.path.join(save_path, "logs")
    model_path = os.path.join(save_path, "model")
    code_path = os.path.join(save_path, "code")
    eval_path = os.path.join(save_path, "eval")
    mkdir(save_path); mkdir(log_path); mkdir(model_path); mkdir(code_path); mkdir(eval_path)

    # summary_writer = SummaryWriter(log_dir=log_path)
    # print("\n\ttensorboard --logdir {} --host 0.0.0.0 --port=<port-number>\n".format(log_path))


    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(os.path.join(log_path, "training_log.txt"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Current code save
    tar = tarfile.open( os.path.join(code_path, 'sources.tar'), 'w' )
    tar.add( 'datasets' )
    tar.add( 'model' )
    curr_file = os.listdir(os.getcwd())
    curr_file = [tar.add(i) for i in curr_file if os.path.isdir(i) is False]
    tar.close()
    with open(os.path.join(code_path,'args.txt'), 'w') as f:
        json.dump(dict(vars(args)), f, indent=2)

    # Learnable parameter setting without backbone
    lparams = []
    for name, p in tsm_tgm_model.named_parameters():
        if ("cnn" not in name) & (name.split(".")[0]!="cnn"):
            lparams.append(p)
        else:
            p.requires_grad = False
        print("\t[Trainable]: {:5s} -> {}".format(str(p.requires_grad), name))

    for name, p in ddm_model.named_parameters():
        if ("cnn" not in name) & (name.split(".")[0]!="cnn"):
            lparams.append(p)
        else:
            p.requires_grad = False
        print("\t[Trainable]: {:5s} -> {}".format(str(p.requires_grad), name))

    # Optimizer definition
    optimizer = torch.optim.Adam([{"params" : lparams}], lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCELoss()

    global_log = {}
    step = 0

    tsm_tgm_model.train()
    ddm_model.train()

    # Train iteration: Change 2000 iteration 60 epoch to 30000 iteration 4 cycle for reproducibility
    for cycle in range(args.cycles):
        loader = DataLoader(train_generator, batch_size=args.batch_size, num_workers=16, shuffle=False)
        p_bar = tqdm.tqdm(loader, unit='iter')

        start = time.time()
        for gi, A, P, N, ED in p_bar:
            A = A.cuda().squeeze(1) if A.ndim==5 else A.cuda()
            P = P.cuda().squeeze(1) if P.ndim==5 else P.cuda()
            N = N.cuda().squeeze(1) if N.ndim==5 else N.cuda()
            ED = ED.cuda().squeeze(1) if ED.ndim==5 else ED.cuda()

            optimizer.zero_grad()

            in_data = {
                "anchor"  : A,
                "positive": P,
                "negative": N,
                "easy_distractor" : ED,
            }
            data_time = time.time() - start
            start = time.time()

            # If easy_distractor loaded, forward with ddm_model
            if len(in_data['easy_distractor'].size()) == 4 and args.vvs_ddm:
                ddm_out = ddm_model(in_data['anchor'], easy_distractor_feature=in_data['easy_distractor']) # DDM forwarding
                # Hadamard product of the confidence W_di and the input features X

                ddm_out['confidence'] = ddm_out['confidence'] / args.vvs_sigmoid_T_ddm
                in_data['anchor'] = ddm_out['features'] * torch.sigmoid(ddm_out['confidence']).unsqueeze(-1).unsqueeze(-1).unsqueeze(0) 

                loss_out = tsm_tgm_model(in_data=in_data, is_anc_processed=True)
            else:
                loss_out = tsm_tgm_model(in_data=in_data, is_anc_processed=False)

            # If easy_distractor loaded, train BCE Loss and Acc
            if len(in_data['easy_distractor'].size()) == 4 and args.vvs_ddm:
                loss_out['Loss_distractor'] = loss_fn(torch.sigmoid(ddm_out['confidence']), ddm_out['pseudo_label']) # L_di: BCE loss between W_di and Y_di
                if loss_out["Loss_total"] is not None:
                    loss_out["Loss_total"] = loss_out["Loss_total"] + loss_out['Loss_distractor'] * args.weight_ddm
                else:
                    loss_out["Loss_total"] = loss_out['Loss_distractor']

            if loss_out["Loss_total"] is not None:
                loss_out["Loss_total"].backward()
                optimizer.step()


            model_time = time.time() - start
            start = time.time()

            step += gi.shape[0]

            # train_writer(summary_writer, loss_out, step, "Batch")
            
            log_time = time.time() - start
            start = time.time()

            time_out = {
                "Time_data"  : torch.tensor(data_time),
                "Time_model" : torch.tensor(model_time),
                "Time_log" : torch.tensor(log_time),
            }

            global_log = global_logger(
                    global_logger(global_log, time_out), 
                loss_out)
            logline = ""
            for k, v in global_log.items():
                spt = k.split("_")
                spt = "_".join([spt[0][0], spt[1][0:2]])
                spt += ":{:5.3f}, ".format(v[0])
                logline += spt
            logline = logline[: -2]
            p_bar.set_description(logline)

            if step % 100 == 0:
                loggingline = "cycle:{:02d}, iter:{:06d}, ".format(cycle, step) + logline
                logger.info(loggingline)
            # Save checkpoint
            if step % 10000 == 0:
                print("")
                state = {
                    "tsm_tgm_model" : tsm_tgm_model.state_dict(),
                    "ddm_model" : ddm_model.state_dict(),
                    "optimizer" : optimizer.state_dict()
                }
                torch.save(state, os.path.join(model_path, 'm{:08d}.pth'.format(step)))
        

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

    torch.use_deterministic_algorithms(False)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
    
    train_func(args)
