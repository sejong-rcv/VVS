import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model.model_utils import *

import os
import h5py
import random


class SelfSimEncoder(nn.Module):
    def __init__(self, args, bottlenect_dim=128):
        super(SelfSimEncoder, self).__init__()
        self.args = args
        self.bottleneck_layer = Bottleneck(1, bottlenect_dim) 
        self.attention_layer = nn.TransformerEncoderLayer(d_model=bottlenect_dim*4, nhead=8, dim_feedforward=bottlenect_dim*8)

    def forward(self, features):
        bottleneck_out = self.bottleneck_layer(features.unsqueeze(0))
        diag_input = torch.diagonal(bottleneck_out, dim1=2, dim2=3).permute(0, 2, 1).contiguous()
        attention_output = self.attention_layer(diag_input)
        self_weights = torch.sigmoid(torch.mean(attention_output, dim=-1).squeeze(0) / self.args.vvs_sigmoid_T_tsm)
        return self_weights
   
   
class VideoEncoder(nn.Module):
    def __init__(self):
        super(VideoEncoder, self).__init__()
        pass
        
    def forward(self, features):
        # Applying S-GAP : t_feats
        # Applying ST-GAP: v_feats
        if features.dim() == 4:
            # B, T, S, C
            t_feats = torch.mean(features, dim=2)
            v_feats = torch.mean(t_feats, dim=1)
            t_feats = F.normalize(t_feats, p=2, dim=-1)
            v_feats = F.normalize(v_feats, p=2, dim=-1) # Normalizing a vector makes its direction negligible

        elif features.dim() == 3:
            t_feats = features
            v_feats = torch.mean(features,dim=1)
            v_feats = F.normalize(v_feats, p=2, dim=-1)
        return t_feats, v_feats # [1, T, 3840], [1, 3840]


class RefinementNetwork(nn.Module):
    def __init__(self, sigmoid_T=512.0, h_connect=True):
        super(RefinementNetwork, self).__init__()
        self.sigmoid_T = sigmoid_T
        self.h_connect = h_connect

        self.cpad1 = nn.ConstantPad1d(1,0)
        self.conv1 = nn.Conv1d(1, 32, 3)

        self.cpad2 = nn.ConstantPad1d(1,0)
        self.conv2 = nn.Conv1d(32, 64, 3)

        self.cpad3 = nn.ConstantPad1d(1,0)
        self.conv3 = nn.Conv1d(64, 128, 3)

        self.fconv = nn.Conv1d(128, 1, 1)
        if self.h_connect:
            self.aconv = nn.Conv1d(225, 1, 1)

    def forward(self, x):
        x = self.cpad1(x)
        x = self.conv1(x)
        x1 = x.clone()
        x = F.relu(x)

        x = self.cpad2(x)
        x = self.conv2(x)
        x2 = x.clone()
        x = F.relu(x)

        x = self.cpad3(x)
        x = self.conv3(x)
        x3 = x.clone()
        x = F.relu(x)

        x = self.fconv(x)

        if self.h_connect:
            # Hierarchical connection in TGM
            x = torch.cat((x1, x2, x3, x), dim=1)
            x = self.aconv(x)

        x = x / self.sigmoid_T
        x = torch.sigmoid(x)

        return x
    

class TopicGuidanceModule(nn.Module):
    def __init__(self, vinit="v_t", sigmoid_T=512.0, h_connect=True, refinement=True):
        super(TopicGuidanceModule, self).__init__()

        self.vinit = vinit
        self.sigmoid_T = sigmoid_T
        self.h_connect = h_connect
        self.refinement = refinement

        self.v_encoder_layer = VideoEncoder()
        self.tensor_dot_v_t = TensorDot("bc,btc->bt")
        self.weight_comperator_layer = RefinementNetwork(sigmoid_T=self.sigmoid_T, h_connect=self.h_connect)

    def forward(self, x):
        t_x, v_x = self.v_encoder_layer(x) # [1, T, 3840], [1, 3840](pseudo topic G)

        v_t = self.tensor_dot_v_t(v_x, t_x).unsqueeze(1) # Get initial state I

        if self.vinit=="const":
            v_t = torch.zeros_like(v_t) + 0.5
        elif self.vinit=="rand":
            v_t = torch.rand_like(v_t)
        elif (self.vinit=="v_t") is False:
            raise ValueError("Check vinit option!")

        if self.refinement:
            w_v_t = self.weight_comperator_layer(v_t)
        else:
            w_v_t = v_t

        if x.dim() == 4:
            weights = w_v_t.squeeze(1).unsqueeze(-1).unsqueeze(-1)
        elif x.dim() == 3:
            weights = w_v_t.squeeze(1).unsqueeze(-1)

        out = {
            "weights" : weights
        }

        return out


class EasyDistractorEliminationStage(nn.Module):
    def __init__(self, args):
        super(EasyDistractorEliminationStage, self).__init__()
        self.backbone_type ='vcdb_' + args.feature_extract
        self.base_normalize = args.base_normalize
        self.pca_whitening = args.pca_whitening
        self.pca_reduction = args.pca_reduction
        self.base_split = [256, 512, 1024, 2048]
        self.gpu = args.gpu
        self.distractor_sampling_ratio = args.distractor_sampling_ratio

        if self.base_normalize:
            print("\t[Option] Base normalize is activated")

        self.pca_layer = PCA(
                n_components=3840 if self.pca_reduction is None else self.pca_reduction,
                whitening=self.pca_whitening)

        self.v_encoder_layer = VideoEncoder()

        self.ddm_layer1 = nn.Linear(3840, 960)
        self.ddm_layer2 = nn.Linear(960, 240)
        self.ddm_layer3 = nn.Linear(240, 1)
        self.ddm_drop = nn.Dropout()
        self.ddm_relu = nn.ReLU()

    def extract_features(self, features, easy_distractor_feature=None):
        # Inject distractors at a predefined ratio
        if (easy_distractor_feature!=None) & self.training:
            if self.distractor_sampling_ratio == '0_20':
                ed_len = random.randint(0, int(features.shape[1]*0.2))
            elif self.distractor_sampling_ratio == '20_50':
                ed_len = random.randint(int(features.shape[1]*0.2), int(features.shape[1]*0.5))
            elif self.distractor_sampling_ratio == '50_80':
                ed_len = random.randint(int(features.shape[1]*0.5), int(features.shape[1]*0.8))
            elif self.distractor_sampling_ratio == '80_100':
                ed_len = random.randint(int(features.shape[1]*0.8), int(features.shape[1]*1.0))
                
            ed_pos = random.randint(0, easy_distractor_feature.shape[1]-ed_len)
            insert_pos = random.randint(0, features.shape[1]-ed_len)

            ed_ind = torch.sort(torch.arange(insert_pos, insert_pos+ed_len)).values
            other_ind = torch.from_numpy(np.setdiff1d(torch.arange(0, features.shape[1]), ed_ind))

            ed_ind = ed_ind.cuda()
            other_ind = other_ind.cuda()

            if insert_pos == 0:
                template = torch.cat((easy_distractor_feature.squeeze(0)[ed_pos:ed_pos+ed_len].unsqueeze(0), features), dim=1)
            elif insert_pos == features.shape[1] - ed_len:
                template = torch.cat((features, easy_distractor_feature.squeeze(0)[ed_pos:ed_pos+ed_len].unsqueeze(0)), dim=1)
            else:
                split_features = torch.split(features, [insert_pos, features.shape[1]-insert_pos], dim=1)
                template = torch.cat((split_features[0],easy_distractor_feature.squeeze(0)[ed_pos:ed_pos+ed_len].unsqueeze(0), split_features[1]), dim=1)
            
            pseudo_label = torch.ones(features.shape[1] + ed_len).cuda()
            pseudo_label[ed_ind] = 0

            features = template   


        # Preprocessing, If features are processed here, they are skipped on the next stage.
        if self.base_normalize: 
            # 3840 = 256 + 512 + 1024 + 2048
            split_features = torch.split(features, self.base_split, dim=-1)
            norm_features = torch.cat([F.normalize(i, p=2, dim=-1) for i in split_features], dim=-1)
            features = F.normalize(norm_features, p=2, dim=-1) 

        if self.pca_whitening or self.pca_reduction:
            pca_features = self.pca_layer(features)
            spooled_features = F.normalize(torch.mean(features, dim=2), p=2, dim=-1) # [T, 9, 3840] -> [T, 3840], S-GAP

        out = {
            "features" : pca_features,
            "spooled_features" : spooled_features
        }

        if (easy_distractor_feature!=None) & self.training:
            out["pseudo_label"] = pseudo_label

        return out

    def distractor_discrimination_module(self, x): # DDM
        # Multiple layers to calculate confidence score W_di
        predict = self.ddm_layer1(x)       # [1, T, 3840] -> [1, T, 960]
        predict = self.ddm_relu(predict)
        predict = self.ddm_drop(predict)
        predict = self.ddm_layer2(predict) # [1, T, 960]  -> [1, T, 240]
        predict = self.ddm_relu(predict)
        predict = self.ddm_drop(predict)
        predict = self.ddm_layer3(predict) # [1, T, 240]  -> [1, T, 1]

        return predict.squeeze(-1).squeeze(0) # [T]

    def forward(self, x, easy_distractor_feature=None):
        features_out = self.extract_features(features=x, easy_distractor_feature=easy_distractor_feature)  # Get injected features and pseudo label Y_di
        confidence = self.distractor_discrimination_module(features_out['spooled_features']) # Get foreground confidence score W_di
        features_out['confidence'] = confidence

        return features_out 
        

class SuppressionWeightGenerationStage(nn.Module):
    def __init__(self, args):
        super(SuppressionWeightGenerationStage, self).__init__()
       
        self.args = args

        self.backbone_type ='vcdb_' + args.feature_extract
        self.base_normalize = args.base_normalize
        self.pca_whitening = args.pca_whitening
        self.pca_reduction = args.pca_reduction
        self.base_split = [256, 512, 1024, 2048]
        self.suppression = args.suppression
        
        # Arguments for ablation studies
        self.vvs_vinit = args.vvs_vinit
        self.vvs_h_connect = args.vvs_h_connect
        self.vvs_sigmoid_T = args.vvs_sigmoid_T
        
        self.refinement = args.refinement
        self.vvs_tsm = args.vvs_tsm
        self.vvs_tgm = args.vvs_tgm
        
        self.weight_triplet = args.weight_triplet
        self.weight_saliency = args.weight_saliency
        self.weight_frame = args.weight_frame
        self.thresh_s = args.thresh_s
        
        if self.base_normalize:
            print("\t[Option] Base normalize is activated")

        self.pca_layer = PCA(
                n_components=3840 if self.pca_reduction is None else self.pca_reduction,
                whitening=self.pca_whitening)
   
        print("\t[Option] Suppression is activated")
        print("\t\t Type: {}".format(self.suppression))
        if self.suppression=="vvs":
            self.topic_guidance_layer = TopicGuidanceModule(
                vinit=self.vvs_vinit, sigmoid_T=self.vvs_sigmoid_T, h_connect=self.vvs_h_connect, refinement=self.refinement)
            print("\t\t\t VVS-VInit: {}".format(self.vvs_vinit))
            print("\t\t\t VVS-SigmoidTemporature: {}".format(self.vvs_sigmoid_T))
            print("\t\t\t VVS-HierarchicalConnection: {}".format(self.vvs_h_connect))
            print("\t\t\t VVS-SaliencyLoss: {}".format(self.vvs_tsm))
        else: 
            raise ValueError("Check suppression option!")

        self.v_encoder_layer = VideoEncoder()
        
        self.tensor_dot = TensorDot("bnc,bnc->bn")
        self.tensor_dot_tc = TensorDot("ic,jc->ij")
        self.tensor_dot_tsc = TensorDot("biok,bjpk->biopj")

        if self.vvs_tsm:
            self.sim_func1 = ChamferSimilarity(axes=[3, 2])
            self.sim_func2 = ChamferSimilarity(axes=[2, 1])
            self.htanh = nn.Hardtanh()
            self.comperator_layer = VideoComperator()

            self.selfsim_encoder = SelfSimEncoder(args)
            self.bceloss = torch.nn.BCELoss()
            self.saliency_thr = nn.parameter.Parameter(torch.tensor(self.thresh_s), requires_grad=False)
        
        self.zero = nn.parameter.Parameter(torch.tensor(0.0), requires_grad=False)
        self.gamma = nn.parameter.Parameter(torch.tensor(0.5), requires_grad=False)

    
    def video_level_encoder(self, in_feats):
        features = in_feats["features"]
        t_x, v_x = self.v_encoder_layer(features)
        out_feats = {
            "features" : v_x
        }
        for k, v in in_feats.items():
            if k not in out_feats:
                out_feats.update({k: v})

        return out_feats

    def preprocessing(self, features):
        if self.base_normalize: 
            # 3840 = 256 + 512 + 1024 + 2048
            split_features = torch.split(features, self.base_split, dim=-1)
            norm_features = torch.cat([F.normalize(i, p=2, dim=-1) for i in split_features], dim=-1)
            features = F.normalize(norm_features, p=2, dim=-1) 

        if self.pca_whitening or self.pca_reduction:
            features = self.pca_layer(features)

        return features
        
    def calculate_pair_sim(self, x, y):
        x_feats = x
        if x_feats.ndim == 2:
            x_feats = x_feats.unsqueeze(1)

        y_feats = y
        if y_feats.ndim == 2:
            y_feats = y_feats.unsqueeze(1)

        x_feats = F.normalize(x_feats, p=2, dim=-1)
        y_feats = F.normalize(y_feats, p=2, dim=-1)
        sim_v = self.tensor_dot(x_feats, y_feats)
        return sim_v
    
    def calculate_triplet_loss(self, pos_sim, neg_sim):
        L_total = 0

        tri_v = neg_sim - pos_sim + self.gamma
        L_v = torch.maximum(self.zero, tri_v)
        L_v = torch.mean(L_v)
        L_total += L_v

        return L_total

    def calculate_frame_sim(self, x, y):
        x_size = x.shape[1]
        y_size = y.shape[1]

        while x.shape[1] < 4:
            x = torch.cat((x, x), dim=1)

        while y.shape[1] < 4:
            y = torch.cat((y, y), dim=1)

        x_resize = x.shape[1]
        y_resize = y.shape[1]
            
        fsim_raw = self.tensor_dot_tsc(x, y)
        fsim = self.sim_func1(fsim_raw) # TD + CS

        vsim_raw = self.comperator_layer(fsim.unsqueeze(1)).squeeze(1)
        vsim = self.sim_func2(self.htanh(vsim_raw)) # CS
        return vsim, vsim_raw, x_size, y_size, x_resize, y_resize

    def sim_reg_loss(self, sim, lower_limit=-1., upper_limit=1.):
        low = torch.abs(torch.minimum(self.zero, sim - lower_limit))
        low_l = torch.sum(low.view(low.shape[0], -1), dim=1)
        high = torch.abs(torch.maximum(self.zero, sim - upper_limit))
        high_l = torch.sum(high.view(high.shape[0], -1), dim=1)

        return (low_l + high_l)[0]

    # Pipeline of TGM
    def topic_guidance_module(self, a_feats, p_feats=None, n_feats=None):
        a_tg_weights = self.topic_guidance_layer(a_feats)["weights"]

        p_tg_weights = None
        if p_feats is not None:
            p_tg_weights = self.topic_guidance_layer(p_feats)["weights"]

        n_tg_weights = None
        if n_feats is not None:
            n_tg_weights = self.topic_guidance_layer(n_feats)["weights"]

        return a_tg_weights, p_tg_weights, n_tg_weights

    # Pipeline of TSM
    def temporal_saliency_module(self, a_feats, p_feats=None, n_feats=None):

        self_sim_raw = self.tensor_dot_tsc(a_feats, a_feats)
        self_sim = self.sim_func1(self_sim_raw)
        ts_weights = self.selfsim_encoder(self_sim) # Saliency weights W_sa
        
        frame_loss = None
        saliency_loss = None
        
        if (p_feats is not None) & (n_feats is not None):

            ap_vsim, ap_vsim_raw, ap_x_size, ap_y_size, ap_x_resize, ap_y_resize  = self.calculate_frame_sim(a_feats, p_feats)
            an_vsim, an_vsim_raw, an_x_size, an_y_size, an_x_resize, an_y_resize = self.calculate_frame_sim(a_feats, n_feats)
            
            anchor_fscore = torch.max(ap_vsim_raw.squeeze(0), dim=-1).values
            max_mask = anchor_fscore == torch.max(anchor_fscore, dim=-1).values
            thre_mask = anchor_fscore > ap_vsim[0]
            saliency_mask = max_mask | thre_mask

            saliency_mask = F.interpolate(saliency_mask.unsqueeze(0).unsqueeze(0).float(), size=ap_x_resize, mode='nearest').squeeze(0).squeeze(0)
            saliency_mask = saliency_mask[:ap_x_size] # Saliency label Y_sa

            saliency_meta = [torch.tensor(int(saliency_mask.sum().item())), torch.tensor(saliency_mask.shape[0])]

            reg_loss = self.sim_reg_loss(ap_vsim_raw) + self.sim_reg_loss(an_vsim_raw) # L_reg
            frame_loss = torch.mean(torch.maximum(self.zero, an_vsim - ap_vsim + self.gamma)) + self.gamma * reg_loss # L_fr = L_tri + gamma*L_reg

            saliency_loss = self.bceloss(ts_weights, saliency_mask) # L_sa: BCE loss between W_sa and Y_sa

        if self.training:
            if self.args.weight_frame != 0.0:
                return ts_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), frame_loss, saliency_loss, (ap_vsim, an_vsim, saliency_meta)
            else:
                return ts_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), frame_loss, saliency_loss, None
        else: 
            return ts_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), frame_loss, saliency_loss, None

    def forward(self, in_data, is_anc_processed=False):
        triplet_loss, frame_loss, saliency_loss, frame_loss_score = None, None, None, None
        pos_sim, neg_sim = None, None
        
        # Feature preprocessing
        if is_anc_processed is True:
            a_feats = in_data["anchor"]
        else:
            a_feats = self.preprocessing(features=in_data["anchor"])
        p_feats = self.preprocessing(features=in_data["positive"]) if "positive" in in_data else None
        n_feats = self.preprocessing(features=in_data["negative"]) if "negative" in in_data else None
    
        # Feature forwarding
        if self.vvs_tgm:
            a_tg_weights, p_tg_weights, n_tg_weights = self.topic_guidance_module(a_feats, p_feats, n_feats)  # TGM forwarding

        else:            
            a_tg_weights, p_tg_weights, n_tg_weights = None, None, None
            
        if self.vvs_tsm:
            a_ts_weights, frame_loss, saliency_loss, frame_loss_score = self.temporal_saliency_module(a_feats, p_feats, n_feats) # TSM forwarding
        else:
            a_ts_weights, frame_loss, saliency_loss, frame_loss_score = None, None, None, None
            
        # Weight generation
        if self.vvs_tgm and self.vvs_tsm:
            a_weights = a_tg_weights *  a_ts_weights
        else: 
            if self.vvs_tgm:
                a_weights = a_tg_weights
            elif self.vvs_tsm:
                a_weights = a_ts_weights
            else:
                a_weights = None
            
        if self.vvs_tgm: 
            p_weights = p_tg_weights
            n_weights = n_tg_weights
            
            _, a_v_feats = self.v_encoder_layer(a_feats * a_weights)
            _, p_v_feats = self.v_encoder_layer(p_feats * p_weights) if "positive" in in_data else (None, None)
            _, n_v_feats = self.v_encoder_layer(n_feats * n_weights) if "negative" in in_data else (None, None)

            pos_sim = self.calculate_pair_sim(a_v_feats, p_v_feats) if "positive" in in_data else None
            neg_sim = self.calculate_pair_sim(a_v_feats, n_v_feats) if "negative" in in_data else None
            
            # L_vi: triplet margin loss between video-level features
            triplet_loss = self.calculate_triplet_loss(pos_sim, neg_sim) if ("positive" in in_data) & ("negative" in in_data) else None
        else:
            if a_weights is not None: # TSM weight

                _, a_v_feats = self.v_encoder_layer(a_feats * a_weights)
            else:
                _, a_v_feats = self.v_encoder_layer(a_feats)
                    
        if (("positive" in in_data) & ("negative" in in_data)): # train mode
            if self.vvs_tgm and self.vvs_tsm:
                total_loss = self.weight_triplet*triplet_loss + self.weight_saliency*saliency_loss + self.weight_frame*frame_loss # Calculate total loss within VVS model
            else:
                if self.vvs_tgm:
                    total_loss = self.weight_triplet*triplet_loss
                elif self.vvs_tsm:
                    total_loss = self.weight_saliency*saliency_loss + self.weight_frame*frame_loss
                else: 
                    # Only DDM is used to filter the frames
                    total_loss = None
        else:
            total_loss = None

        out = {
            "features" : a_v_feats,
            "Loss_triplet": triplet_loss, 
            "Loss_total": total_loss
        }

        if saliency_loss is not None:
            out["Loss_saliency"] = saliency_loss
        if frame_loss is not None:
            out["Loss_frame"] = frame_loss
        if frame_loss_score is not None:
            out["Loss_fp"] = frame_loss_score[0][0]
            out["Loss_fn"] = frame_loss_score[1][0]
            out["Loss_sg"] = frame_loss_score[2][0]
            out["Loss_sn"] = frame_loss_score[2][1]
            if pos_sim is not None:
                out["Loss_vp"] = pos_sim[0][0]
            if neg_sim is not None:
                out["Loss_vn"] = neg_sim[0][0]
            

        return out
