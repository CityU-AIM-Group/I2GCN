import os 
import torch 
import torch.nn as nn 
import time 
import json 

import config
from utils import print_cz, time_mark, expand_user, seed_fix
from train_func import train
import dataloader_npy
from model import resnet, resnet_in
from model import agg_fc, agg_i2gcn

def prepare():
    """
        config, make dirs and logger file
    """
    args = config.get_args()
    time_tag = time_mark()
    log_dir = config.save_dir + time_tag + '-' + args.theme  + '-testsplit'+str(args.test_split)+'_' + args.optim + '_lr'+str(args.lr) + '_step'+str(args.lr_step) +'_gamma'+str(args.lr_gamma) + '_bs'+str(args.batch_size) + '_epochs'+str(args.epochs)

    if os.path.exists(log_dir) is False:# make dir if not exist
        os.makedirs(expand_user(log_dir))
        print('make dir: ' + str(log_dir))
    return args, log_dir

def main():
    print('start...')
    args, log_dir = prepare()
    log_file = open((log_dir + '/' + 'print_out_screen.txt'), 'w')
    print_cz("===> Preparing", f=log_file)
    t = time.time()
    #
    seed_fix(args, logfile=log_file)
    
    print_cz("===> Building model", f=log_file)
    if args.normalization_flag in ['IN', 'in']:
        extractor = resnet_in.resnet18(pretrained=bool(args.pretrained_flag), instance_affine=False)
    elif args.normalization_flag in ['BN', 'bn']:
        extractor = resnet.resnet18(pretrained=bool(args.pretrained_flag))
    else:
        print_cz('Error normalization flag')
    aggregator_pre = agg_fc.Aggregator(class_num=3, in_dim=512, mean_flag=True) 
    aggregator = agg_i2gcn.I2GCN(
        class_num=3, 
        in_dim=512, 
        inter_dim=args.inter_dim, 
        out_dim=args.out_dim, 
        node_num=16,
        sigma=args.sigma,
        adj_ratio=args.adj_ratio,
        self_loop_flag=True,
        keep_top=args.keep_top, 
        drop_p=args.drop_p,
        gc_bias=bool(args.gcn_bias)
    )
    
    criterion = nn.CrossEntropyLoss()

    print_cz("===> Setting GPU", f=log_file)
    if args.job_type == 'S' or args.job_type == 's':
        extractor = extractor.cuda()
        aggregator_pre = aggregator_pre.cuda()
        aggregator = aggregator.cuda()
    else:
        if args.job_type == 'Q' or args.job_type == 'q':
            gpu_device_ids=[0, 1, 2, 3]
        elif args.job_type == 'E' or args.job_type == 'e':
            gpu_device_ids=[0, 1, 2, 3, 4, 5, 6, 7]
        elif args.job_type == 'D' or args.job_type == 'd':
            gpu_device_ids=[0, 1]
        extractor = nn.DataParallel(extractor.cuda(), device_ids=gpu_device_ids).cuda()
        aggregator_pre = nn.DataParallel(aggregator_pre.cuda(), device_ids=gpu_device_ids).cuda()
        aggregator = nn.DataParallel(aggregator.cuda(), device_ids=gpu_device_ids).cuda()

    
    print_cz("===> Loading datasets", f=log_file)  
    train_loader = dataloader_npy.get_dataloader(
        online_flag=bool(args.online_flag), 
        folder_public=config.data_public_dir, 
        test_split=args.test_split, 
        bsz=args.batch_size, 
        num_workers=args.num_workers, 
        stage='train', 
        scan_shuffle=True,  
        augmentations=True,
        )
    valid_loader = dataloader_npy.get_dataloader(
        online_flag=bool(args.online_flag), 
        folder_public=config.data_public_dir, 
        test_split=args.test_split,
        bsz=args.batch_size, 
        num_workers=args.num_workers, 
        stage='valid', 
        scan_shuffle=False,  
        augmentations=False,
        )
    test_loader  = dataloader_npy.get_dataloader(
        online_flag=bool(args.online_flag), 
        folder_public=config.data_public_dir, 
        test_split=args.test_split,
        bsz=args.batch_size, 
        num_workers=args.num_workers, 
        stage='test', 
        scan_shuffle=False,  
        augmentations=False,
        )

    print_cz("===> Setting Optimizer", f=log_file)
    if args.optim in ['Adam', 'adam']:
        optimizer_extractor = torch.optim.Adam(params=extractor.parameters(), lr=args.lr*args.lr_factor, weight_decay=args.wd)
        optimizer_aggregator_pre = torch.optim.Adam(params=aggregator_pre.parameters(), lr=args.lr, weight_decay=args.wd)
        optimizer_aggregator = torch.optim.Adam(params=aggregator.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim in ['SGD', 'sgd']:
        optimizer_extractor = torch.optim.SGD(params=extractor.parameters(), lr=args.lr*args.lr_factor, weight_decay=args.wd, momentum=args.momen)
        optimizer_aggregator_pre = torch.optim.SGD(params=aggregator_pre.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momen)
        optimizer_aggregator = torch.optim.SGD(params=aggregator.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momen)
    
    print_cz("===> Training", f=log_file)
    train(
        train_loader=train_loader, 
        valid_loader=valid_loader,
        test_loader=test_loader, 
        extractor=extractor,
        aggregator=aggregator,
        aggregator_pre=aggregator_pre,
        criterion=criterion, 
        optimizer_extractor=optimizer_extractor, 
        optimizer_aggregator=optimizer_aggregator, 
        optimizer_aggregator_pre=optimizer_aggregator_pre, 
        args=args,
        logfile=log_file, 
        save_path=log_dir+'/', 
        )

    print_cz('Total time: {:.2f} h'.format((time.time()-t)/3600.0), f=log_file)
    log_file.close()

if __name__ == '__main__':

    main()