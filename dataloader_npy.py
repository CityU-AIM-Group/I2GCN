import os
import sys 
from PIL import Image
import random
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import pandas as pd
import csv
import time
import config


class CT_Dataset_online(data.Dataset):
    def __init__(self, folder_public, csv_file, augmentations, pre_size=420, final_size=384):
        super(CT_Dataset_online, self).__init__()   
        self.folder_public = folder_public ###
        self.augmentations = augmentations
        # color augmentation
        self.RANDOM_BRIGHTNESS = 7
        self.RANDOM_CONTRAST = 5
        self.pre_size = pre_size
        self.final_size = final_size
        self.spatial_limit = int((pre_size-final_size)/2.0)
        self.pre_top_left = int((512-self.pre_size)/2.0)
        self.final_top_left = int((512-self.final_size)/2.0)

        if isinstance(csv_file, str):
            df = pd.read_csv(os.path.join(config.csv_dir, csv_file))
        else:
            print('csv file name error!')
        self.patients = df['patient_id'] 
        self.scans = df['scan_id'] 
        self.targets = df['target']

    def __getitem__(self, index):
        target = int(self.targets[index])
        npy = np.load(
            os.path.join(
                self.folder_public, 
                'p'+str(self.patients[index])+'-s'+str(self.scans[index])+'.npy'
                )
            )
        
        npy_normalized = npy.astype(np.float32) / 255.0 # cast to float
        if self.augmentations:
            # random flip
            if random.uniform(0, 1) < 0.5: #horizontal flip
                npy_normalized = np.flipud(npy_normalized)
            # color jitter
            br = random.randint(-self.RANDOM_BRIGHTNESS, self.RANDOM_BRIGHTNESS) / 100.
            npy_normalized = npy_normalized + br
            # Random contrast
            cr = 1.0 + random.randint(-self.RANDOM_CONTRAST, self.RANDOM_CONTRAST) / 100.
            npy_normalized = npy_normalized * cr
            # clip values to 0-1 range
            npy_normalized = np.clip(npy_normalized, 0, 1.0)
            # random crop
            offset_x = random.randint(-self.spatial_limit, self.spatial_limit)
            offset_y = random.randint(-self.spatial_limit, self.spatial_limit)
            npy_normalized = npy_normalized[
                :, 
                self.final_top_left+offset_x : self.final_top_left+self.final_size+offset_x, 
                self.final_top_left+offset_y : self.final_top_left+self.final_size+offset_y
                ]
        else:
            npy_normalized = npy_normalized[
                :, 
                self.final_top_left : self.final_top_left+self.final_size, 
                self.final_top_left : self.final_top_left+self.final_size
                ]
        return npy_normalized, target, 'p{}-s{}'.format(str(self.patients[index]), str(self.scans[index]))

    def __len__(self):
        return len(self.targets)

class CT_Dataset_offline(data.Dataset):
    def __init__(self, folder_public, csv_file, augmentations, pre_size=420, final_size=384):
        super(CT_Dataset_offline, self).__init__()
        self.folder_public = folder_public ###
        self.augmentations = augmentations
        # color augmentation
        self.RANDOM_BRIGHTNESS = 7
        self.RANDOM_CONTRAST = 5
        self.pre_size = pre_size
        self.final_size = final_size
        self.spatial_limit = int((pre_size-final_size)/2.0)
        self.pre_top_left = int((512-self.pre_size)/2.0)
        self.final_top_left = int((512-self.final_size)/2.0)

        if isinstance(csv_file, str):
            df = pd.read_csv(os.path.join(config.csv_dir, csv_file))
        else:
            print('csv file name error!')
        self.patients = df['patient_id']
        self.scans = df['scan_id'] 
        self.targets = df['target']

        start_time = time.time()
        self.offline_dict = {}
        if isinstance(csv_file, str):
            with open(os.path.join(config.csv_dir, csv_file), 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    zip_file,target,label,patient_id,scan_id,n_slice,scan_count,all_scan_ids = row 
                    self.offline_dict['{}-{}'.format(patient_id, scan_id)] = np.load(
                                                                                os.path.join(
                                                                                    self.folder_public, 
                                                                                    'p'+str(patient_id)+'-s'+str(scan_id)+'.npy'
                                                                                    )
                                                                                )
        else:
            print('csv file name error!')

        print('offline load time:\t{:.2f} min.'.format((time.time()-start_time)/60.0))

    def __getitem__(self, index):
        
        target = int(self.targets[index])
        npy = self.offline_dict['{}-{}'.format(str(self.patients[index]), str(self.scans[index]))]
        
        npy_normalized = npy.astype(np.float32) / 255.0 # cast to float
        if self.augmentations:
            # random flip
            if random.uniform(0, 1) < 0.5: # 
                npy_normalized = np.flipud(npy_normalized)
            # color jitter
            br = random.randint(-self.RANDOM_BRIGHTNESS, self.RANDOM_BRIGHTNESS) / 100.
            npy_normalized = npy_normalized + br
            # Random contrast
            cr = 1.0 + random.randint(-self.RANDOM_CONTRAST, self.RANDOM_CONTRAST) / 100.
            npy_normalized = npy_normalized * cr
            # clip values to 0-1 range
            npy_normalized = np.clip(npy_normalized, 0, 1.0)

            offset_x = random.randint(-self.spatial_limit, self.spatial_limit)
            offset_y = random.randint(-self.spatial_limit, self.spatial_limit)
            npy_normalized = npy_normalized[
                :, 
                self.final_top_left+offset_x : self.final_top_left+self.final_size+offset_x, 
                self.final_top_left+offset_y : self.final_top_left+self.final_size+offset_y
                ]
        else:
            npy_normalized = npy_normalized[
                :, 
                self.final_top_left : self.final_top_left+self.final_size, 
                self.final_top_left : self.final_top_left+self.final_size
                ]

        return npy_normalized, target, 'p{}-s{}'.format(str(self.patients[index]), str(self.scans[index]))
    def __len__(self):
        return len(self.targets)


def get_dataloader(
    online_flag, 
    folder_public, 
    test_split, 
    stage, 
    bsz, 
    num_workers, 
    scan_shuffle, 
    augmentations, 
    ):
    if stage in ['train', 'Train']:
        csv_file = 'split{:d}_shuffle_train.csv'.format(test_split)
        if online_flag:
            dataset = CT_Dataset_online(folder_public=folder_public, csv_file=csv_file, augmentations=augmentations)
        else:
            dataset = CT_Dataset_offline(folder_public=folder_public, csv_file=csv_file, augmentations=augmentations)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=scan_shuffle, num_workers=num_workers, pin_memory=False)
    #
    elif stage in ['valid', 'Valid']:
        csv_file = 'split{:d}_shuffle_valid.csv'.format(test_split)
        if online_flag:
            dataset = CT_Dataset_online(folder_public=folder_public, csv_file=csv_file, augmentations=False)
        else:
            dataset = CT_Dataset_offline(folder_public=folder_public, csv_file=csv_file, augmentations=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=False, num_workers=num_workers, pin_memory=False)
    #
    elif stage in ['test', 'Test']:
        csv_file = 'split{:d}_shuffle_test.csv'.format(test_split)
        if online_flag:
            dataset = CT_Dataset_online(folder_public=folder_public, csv_file=csv_file, augmentations=False)
        else:
            dataset = CT_Dataset_offline(folder_public=folder_public, csv_file=csv_file, augmentations=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=False, num_workers=num_workers, pin_memory=False)
    #
    print('stage: {},\t csv_name: {}'.format(stage, csv_file))
    print('build online?{} CT test  dataset with {:d} num_workers, batch size {:d}, iters {:d}.'.format(str(online_flag)[0], num_workers, bsz, len(dataloader)))
    return dataloader

if __name__ == '__main__':

    pass 