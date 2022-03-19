import os 
import torch
import argparse

# code_dir = './I2GCN/'
# save_dir = code_dir + 'experiment/'
# csv_dir = code_dir + 'csv'

save_dir = './experiment/'
csv_dir = './csv/'
data_public_dir = './data/'

def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_split", type=int, default=0)
    parser.add_argument("--online_flag", type=int, default=1)
    
    parser.add_argument("--optim", type=str, default="Adam")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_factor", type=float, default=0.1)
    parser.add_argument("--lr_step", type=int, default=50)
    parser.add_argument("--lr_gamma", type=float, default=0.5)

    parser.add_argument("--ratio_pre", type=float, default=1.0)
    parser.add_argument("--ratio_dc", type=float, default=1.0)
    parser.add_argument("--ratio_HSIC", type=float, default=1.0)
    parser.add_argument("--ratio_rank", type=float, default=1.0)
    parser.add_argument("--rank_m", type=float, default=0.05)

    parser.add_argument("--inter_dim", type=int, default=512)
    parser.add_argument("--out_dim", type=int, default=512)
    # parser.add_argument("--neighbor", type=int, default=-1)
    parser.add_argument("--sigma", type=int, default=2)
    parser.add_argument("--adj_ratio", type=float, default=0.2)
    parser.add_argument("--keep_top", type=float, default=0.8)
    parser.add_argument("--drop_p", type=float, default=0.5)
  
    parser.add_argument("--gcn_bias", type=int, default=0)
    parser.add_argument("--ttv_loop", type=int, default=0)
    
    parser.add_argument("--pretrained_flag", type=int, default=0)
    parser.add_argument("--normalization_flag", type=str, default='IN')
    parser.add_argument("--seed_idx", type=int, default=1234)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--momen", type=float, default=0.9)
    parser.add_argument("--log_path", type=str, default=save_dir)
    parser.add_argument("--theme", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--job_type", type=str, default='S')

    parser.add_argument("--mean_flag", type=int, default=1)


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    import json
    args = get_args()
    print(args.__dict__)
    print(type(args.__dict__))
    with open('./args.json', 'w') as f:
        f.write(json.dumps(args.__dict__, indent=4))