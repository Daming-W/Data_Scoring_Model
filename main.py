import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from utils.model import SiameseNetwork
from utils.dataset import pairwise_dataset

parser = argparse.ArgumentParser()
# path and dir
parser.add_argument("--model_save_path", type=str, 
                    default='/root/autodl-tmp/')
# device
parser.add_argument("--gpu_id", type=str, default='cuda',
                    help="GPU id to work on, \'cpu\'.")
# data
parser.add_argument("--train_val_ratio", type=float,
                    default=0.75, help="train set and val set dataset ratio")
parser.add_argument("--batch_size", type=int,
                    default=32, help="batch size of data")      
parser.add_argument("--num_workers", type=int,
                    default=18, help="number of workers")   
# hyperparameters
parser.add_argument("--lr", type=float,
                    default=5e-7, help="learning rate")   
parser.add_argument("--weight_decay", type=float,
                    default=0.001, help="weight_decay")
parser.add_argument("--eps", type=float,
                    default=1e-8, help="eps")
parser.add_argument("--warmup_ratio", type=float, default=0.1,
                    help="accumuate gradients")
parser.add_argument("--epochs", type=int, default=40,
                    help="total number of training epochs")
parser.add_argument("--resume", type=bool, default=False,
                    help="flag to resume training")
args = parser.parse_args()


if __name__=="__main__":
    # get model
    model = SiameseNetwork()
    dataset = pairwise_dataset(args, args.path)

