from operator import ne
import numpy as np
import time
from tqdm import tqdm
from imgaug import augmenters as iaa
from numpy.random import uniform as rnd
from utils import load_video, slow_video_load


class Augmentor(object):
    def __init__(self, videos, masks, window):
        self.video_paths = videos
        self.window = window

        # Load VCDB masks
        self.masks = dict()
        for k, v in tqdm(masks.items()):
            try:
                self.masks[k] = np.load(v)
            except:
                print("[Error] Failed to load mask : ", v)

    def prepare_triplet(self, anchor_id, positive_id, negative_id, augmentation=True, fixed_aug=None):
        # Save augmented information due to reproductivity (check below)
        rnd = np.random.uniform() if fixed_aug == None else fixed_aug['rnd']
        augmentation_info = {
            'rnd' : float(rnd),
            'slowmotion_fastforward_augmentation': {
                'slow_mo' : None,
                'fast_fo' : None
            },
            'temporal_augmentation' : {
                'random_temoporal_crop' : None, # start
                'select_window' : [None, None], # start_q / start_p
                'temporal_transformations' : [None, None], # start_q / start_p
                'options' : {
                    'A' : None,
                    'B' : None,
                    'C' : [None, None], # option_C_dur / option_C_rule_0
                    'D' : None
                }
            },
            'spatial_augmentation' : {
                'random_spatial_crop' : [None, None], # top / left
                'trans_random' : None,
                'transforms' : None, # spatial_augmentation
                'col_random' : None,
                'colour' : None # spatial_augmentation
            }
        }

        if not augmentation:
            # Triplet from pool1 (real triplet)
            anchor = load_video(self.video_paths[anchor_id], rs_size=256)
            positive = load_video(self.video_paths[positive_id], rs_size=256)
            negative = load_video(self.video_paths[negative_id], rs_size=256)
            
            mask = self.masks['_'.join(map(str, sorted([anchor_id, positive_id])))]
            if anchor_id > positive_id:
                mask = mask.T

            # If the shape does not match, match it to the format of the loaded video.
            if anchor.shape[0] > mask.shape[0]:
                anchor = anchor[:mask.shape[0]]
            if positive.shape[0] > mask.shape[1]:
                positive = positive[:mask.shape[1]]
        else:
            # Triplet from pool2 (artificial triplet)
            anchor = load_video(self.video_paths[anchor_id], rs_size=256)
            
            mask = np.diag(np.ones((anchor.shape[0]))) # Generate mask for the artificial positive pair
            
            if anchor.shape[0] == 0:
                return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
            
            # Apply slow/fast augmentation
            positive, mask, slow_mo, fast_fo = self.slowmotion_fastforward_augmentation(anchor, anchor_id, mask, rnd, fixed_aug)
            augmentation_info['slowmotion_fastforward_augmentation']['slow_mo'] = str(slow_mo)
            augmentation_info['slowmotion_fastforward_augmentation']['fast_fo'] = str(fast_fo)
            
            negative = load_video(self.video_paths[negative_id], rs_size=256)

        if negative.shape[0] == 0 or anchor.shape[0] != mask.shape[0] or positive.shape[0] != mask.shape[1]:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
        # Apply temporal augmentation
        anchor, positive, negative, mask, tem_aug_info = \
            self.temporal_augmentation(anchor, positive, negative, mask, rnd, augmentation, fixed_aug)
        augmentation_info['temporal_augmentation'] = tem_aug_info

        if negative.shape[0] == 0 or anchor.shape[0] != mask.shape[0] or positive.shape[0] != mask.shape[1]:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
        # Apply spatial transformations if the triplet is artificially generated
        if augmentation:
            positive, spatial_augmentation_info = self.spatial_augmentation(positive, fixed_aug)
            augmentation_info['spatial_augmentation'] = spatial_augmentation_info

        return anchor, positive, negative, mask, augmentation_info

    def temporal_augmentation(self, anchor, positive, negative, mask, rnd, augmentation, fixed_aug=None):
        tem_aug_info = {
            'random_temoporal_crop' : None, # start
            'select_window' : [None, None], # start_q / start_p
            'temporal_transformations' : [None, None], # start_q / start_p
            'options' : {
                'A' : None,
                'B' : None,
                'C' : [None, None], # option_C_dur / option_C_rule_0
                'D' : None
            }
        }
        
        # Repeat videos if they are too short to fit in the selected window
        anchor, mask = self.duplicate_video(anchor, mask=mask, axis=0)
        positive, mask = self.duplicate_video(positive, mask=mask, axis=1)
        negative, start = self.random_temoporal_crop(self.duplicate_video(negative)[0], fixed_aug)
        tem_aug_info['random_temoporal_crop'] = int(start)

        if anchor.shape[0] <= self.window or positive.shape[0] <= self.window:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        # Trim videos based on the selected window
        mask, start_q, start_p, mask_flag = self.select_window(mask, fixed_aug)
        tem_aug_info['select_window'] = [int(start_q), int(start_p)]
        tem_aug_info['mask_flag'] = mask_flag

        anchor = anchor[start_q: start_q + self.window]
        positive = positive[start_p: start_p + self.window]

        mask_anc = [np.argmax(mask.max(1)), self.window - np.argmax(mask.max(1)[::-1])]
        mask_pos = [np.argmax(mask.max(0)), self.window - np.argmax(mask.max(0)[::-1])]

        # Apply other temporal transformations if the triplet is artificially generated
        if augmentation and rnd < 0.0:
            start_p = fixed_aug['temporal_augmentation']['temporal_transformations'][1] \
                    if fixed_aug != None else np.minimum(np.random.randint(mask_pos[0], mask_pos[1]), self.window - 1)
            option_A_range = fixed_aug['temporal_augmentation']['options']['A'] \
                            if fixed_aug != None else np.random.randint(5, self.window / 3)
            tem_aug_info['temporal_transformations'][1] = int(start_p)
            tem_aug_info['options']['A'] = int(option_A_range)

            for i in range(option_A_range):
                positive = np.insert(positive, start_p, positive[start_p], axis=0)
                mask = np.insert(mask, start_p, mask[:, start_p], axis=1)

        elif augmentation and rnd < 0.1:
            start_q = fixed_aug['temporal_augmentation']['temporal_transformations'][0] \
                    if fixed_aug != None else np.minimum(np.random.randint(mask_anc[0], mask_anc[1]), self.window - 1)
            option_B_range = fixed_aug['temporal_augmentation']['options']['B'] \
                    if fixed_aug != None else np.random.randint(0, 10)

            tem_aug_info['temporal_transformations'][0] = int(start_q)
            tem_aug_info['options']['B'] = int(option_B_range)

            for i in range(option_B_range):
                anchor = np.insert(anchor, start_q, anchor[start_q], axis=0)
                mask = np.insert(mask, start_q, mask[start_q], axis=0)

        elif augmentation and rnd < 0.3:
            positive = np.copy(negative)

            dur = fixed_aug['temporal_augmentation']['options']['C'][0] \
                if fixed_aug != None else np.random.randint(5, np.maximum(6, self.window / 4))
            start_q = fixed_aug['temporal_augmentation']['temporal_transformations'][0] \
                if fixed_aug != None else np.random.randint(0, self.window - dur)
            start_p = fixed_aug['temporal_augmentation']['temporal_transformations'][1] \
                if fixed_aug != None else np.random.randint(0, self.window - dur)
                
            tem_aug_info['options']['C'][0] = int(dur)
            tem_aug_info['temporal_transformations'][0] = int(start_q)
            tem_aug_info['temporal_transformations'][1] = int(start_p)

            H_min = np.minimum(anchor.shape[1], positive.shape[1])
            W_min = np.minimum(anchor.shape[2], positive.shape[2])
            positive = positive[:, :H_min, :W_min, :]

            option_C_rule_0 = fixed_aug['temporal_augmentation']['options']['C'][1] \
                            if fixed_aug != None else np.random.uniform()
            tem_aug_info['options']['C'][1] = float(option_C_rule_0)

            if option_C_rule_0 < 0.5:
                segment = anchor[start_q: start_q + dur, :H_min, :W_min, :]
                positive = np.insert(positive, start_p, segment, axis=0)
                mask = np.zeros((self.window, self.window))
                for i in range(dur):
                    mask[start_q + i, start_p + i] = 1.0
            else:
                segment = np.repeat(anchor[start_q:start_q + 1, :H_min, :W_min, :], dur, axis=0)
                positive = np.insert(positive, start_p, segment, axis=0)
                mask = np.zeros((self.window, self.window))
                for i in range(dur):
                    mask[start_q, start_p + i] = 1.0
        
        option_D_rule_0 = fixed_aug['temporal_augmentation']['options']['D'] \
                         if fixed_aug != None else np.random.uniform()
        tem_aug_info['options']['D'] = float(option_D_rule_0)

        if option_D_rule_0 < 0.2:
            positive = positive[::-1]
            mask = mask[:, ::-1]

        mask = mask[:self.window, :self.window]
        anchor = anchor[:self.window]
        positive = positive[:self.window]

        return anchor, positive, negative, mask, tem_aug_info

    def select_window(self, mask, fixed_aug=None):
        if fixed_aug != None:
            mask_flag = fixed_aug['temporal_augmentation']['mask_flag']
            start_q = int(fixed_aug['temporal_augmentation']['select_window'][0])
            start_p = int(fixed_aug['temporal_augmentation']['select_window'][1])
            if mask_flag:
                mask = mask[start_q: start_q + self.window, start_p: start_p + self.window]
        else:
            mask_flag = True
            i = 0
            start_q, start_p = 0, 0
            while i < 10000:
                start_q = np.random.randint(mask.shape[0] - self.window)
                start_p = np.random.randint(mask.shape[1] - self.window)

                m = mask[start_q: start_q + self.window, start_p: start_p + self.window]
                if m.shape[0] == self.window and m.shape[1] == self.window:
                    if m.max(1).sum() > 5:
                        mask = m
                        break
                i += 1
                
            if i >= 10000:
                mask_flag = False
        return mask, start_q, start_p, mask_flag

    def duplicate_video(self, tensor, mask=None, axis=None):
        count = 0
        while tensor.shape[0] <= self.window:
            tensor = np.concatenate([tensor, tensor], axis=0)
            if mask is not None:
                mask = np.concatenate([mask, mask], axis=axis)
            count += 1
            if count >= 64: # self.window
                return tensor, mask
        return tensor, mask

    def random_temoporal_crop(self, tensor, fixed_aug=None):
        start = int(fixed_aug['temporal_augmentation']['random_temoporal_crop']) \
            if fixed_aug != None else np.random.randint(tensor.shape[0] - self.window)
        tensor = tensor[start:start + self.window]
        return tensor, start

    def slowmotion_fastforward_augmentation(self, video, video_id, mask, rnd, fixed_aug=None):
        slow_mo = None 
        fast_fo = None
        
        if 0.9 < rnd:
            if fixed_aug != None:
                slow_mo = int(fixed_aug['slowmotion_fastforward_augmentation']['slow_mo'])
            else:
                slow_mo = np.random.randint(3, 5)
                
            for i in range(video.shape[0]):
                mask = np.insert(mask, i + (slow_mo - 1) * i, np.zeros((slow_mo - 1, mask.shape[0])), axis=1)
                
            # Load video of the same size as the artistically generated mask in order for slowmotion augmentation to work.
            video = slow_video_load(self.video_paths[video_id], rs_size=256, mask_length=mask.shape[1], original_video_length=video.shape[0])
            mask = mask[:, :video.shape[0]]
        elif 0.8 < rnd:
            if fixed_aug != None:
                fast_fo = int(fixed_aug['slowmotion_fastforward_augmentation']['fast_fo'])
            else:
                fast_fo = np.random.randint(2, 4)

            video = video[::fast_fo]
            mask = mask[:, ::fast_fo]

        return video, mask, slow_mo, fast_fo

    def spatial_augmentation(self, video, fixed_aug=None):

        if fixed_aug != None:
            trans_random = fixed_aug['spatial_augmentation']['trans_random']
            col_random = fixed_aug['spatial_augmentation']['col_random']
        else:
            trans_random = [
                float(rnd(0.5, 1.5)), 
                float(rnd(0.5, 1.5)),
                float(rnd(360))
            ]
            col_random = [
                float(rnd(0.7, 1.3)), float(rnd(0.7, 1.3)), float(rnd(0.7, 1.3)), float(rnd(0.7, 1.3)),
                float(rnd(0.7, 1.3)), float(rnd(0.7, 1.3)), float(rnd(0.7, 1.3)), float(rnd(0.7, 1.3)),
                float(rnd(0.7, 1.3)), float(rnd(0.7, 1.3)), float(rnd(0.7, 1.3)), float(rnd(0.7, 1.3))
            ]

        transforms = [
            iaa.Affine(scale={"x": trans_random[0], "y": trans_random[1]}, random_state=0).to_deterministic(),
            iaa.Affine(rotate=trans_random[2], random_state=0).to_deterministic(),
            iaa.Fliplr(1, random_state=0).to_deterministic(),
            iaa.Flipud(1, random_state=0).to_deterministic(),
            iaa.Rot90(1, keep_size=False, random_state=0).to_deterministic(),
            iaa.Rot90(3, keep_size=False, random_state=0).to_deterministic()
        ]

        colour = np.array([
            iaa.Sequential([
                iaa.MultiplyHueAndSaturation(mul_hue=col_random[0], mul_saturation=col_random[1], random_state=0).to_deterministic(),
                iaa.MultiplyBrightness(col_random[2], random_state=0).to_deterministic(), 
                iaa.GammaContrast(col_random[3], random_state=0).to_deterministic()]).to_deterministic(),
            iaa.Sequential([
                iaa.MultiplyBrightness(col_random[4], random_state=0).to_deterministic(), 
                iaa.GammaContrast(col_random[5], random_state=0).to_deterministic(),
                iaa.MultiplyHueAndSaturation(mul_hue=col_random[6], mul_saturation=col_random[7], random_state=0).to_deterministic()]).to_deterministic(),
            iaa.Sequential([
                iaa.MultiplyBrightness(col_random[8], random_state=0).to_deterministic(),
                iaa.MultiplyHueAndSaturation(mul_hue=col_random[9], mul_saturation=col_random[10], random_state=0).to_deterministic(),
                iaa.GammaContrast(col_random[11], random_state=0).to_deterministic()]).to_deterministic(),
            iaa.Grayscale(alpha=1.0, random_state=0).to_deterministic(), 
            iaa.Identity(random_state=0).to_deterministic()
        ], dtype=object)
        video, top, left = self.random_spatial_crop(video, 256, fixed_aug)
        
        t_choice = int(fixed_aug['spatial_augmentation']['transforms']) \
                if fixed_aug != None else np.random.choice([0,1,2,3,4,5])
        c_choice = int(fixed_aug['spatial_augmentation']['colour']) \
                if fixed_aug != None else np.random.choice([0,1,2,3,4])

        seq = iaa.Sequential([transforms[t_choice], colour[c_choice]]).to_deterministic()
        
        spatial_augmentation_info = {
            "random_spatial_crop" : [top, left],
            "trans_random" : trans_random,
            "transforms" : str(t_choice),
            "col_random" : col_random,
            "colour" : str(c_choice)
        }

        return np.array(seq(images=video)), spatial_augmentation_info

    def random_spatial_crop(self, video, desired_size=256, fixed_aug=None):
        _, h, w, _ = video.shape
        if fixed_aug != None:
            top = int(fixed_aug['spatial_augmentation']['random_spatial_crop'][0])
            left = int(fixed_aug['spatial_augmentation']['random_spatial_crop'][1])
        else:
            top = np.random.randint(h - desired_size) if h > desired_size else 0
            left = np.random.randint(w - desired_size) if w > desired_size else 0
        return video[:, top: top + desired_size, left: left + desired_size, :], top, left
