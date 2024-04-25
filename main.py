import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, ConcatDataset
import torch.optim as optim

import torch.multiprocessing as mp

from utils.model import ScoreModel
from utils.dataset import PairDataset
from utils.engine import train_epoch, evaluate_epoch

parser = argparse.ArgumentParser()
# path and dir
parser.add_argument("--model_save_path", type=str, 
                    default='/root/autodl-tmp/')
# device
parser.add_argument("--gpu_id", type=str, default='cuda',
                    help="GPU id to work on, \'cpu\'.")
# data
parser.add_argument("--train_val_ratio", type=float,
                    default=0.8, help="train set and val set dataset ratio")
parser.add_argument("--batch_size", type=int,
                    default=32, help="batch size of data")      
parser.add_argument("--num_workers", type=int,
                    default=64, help="number of workers")   
parser.add_argument("--csv_path", type=str,
                    default='/root/Data_Scoring_Model/data/pair15w.csv', help="path of pairwise dataset csv")   
# hyperparameters
parser.add_argument("--lr", type=float,
                    default=1e-6, help="learning rate")   
parser.add_argument("--weight_decay", type=float,
                    default=0.001, help="weight_decay")
parser.add_argument("--eps", type=float,
                    default=1e-8, help="eps")
parser.add_argument("--warmup_ratio", type=float, default=0.1,
                    help="accumuate gradients")
parser.add_argument("--epochs", type=int, default=10,
                    help="total number of training epochs")
parser.add_argument("--resume", type=bool, default=False,
                    help="flag to resume training")
args = parser.parse_args()


if __name__=="__main__":
    mp.set_start_method('spawn')
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("Using GPU: ", torch.cuda.get_device_name(0))
    torch.cuda.set_device(device)

    # get model
    model = ScoreModel()
    # make sure model params require grad
    for param in model.parameters():
        param.requires_grad = True
    # model to device
    model = model.cuda()

    #load data
    train_dataset = PairDataset(csv_file=args.csv_path, train=True, split_ratio=args.train_val_ratio, transform=None)
    val_dataset = PairDataset(csv_file=args.csv_path, train=False, split_ratio=args.train_val_ratio, transform=None)

    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)
    val_dataloader = DataLoader(val_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)
    print(f'finish loading data {len(train_dataset),len(val_dataset)}')
    
    # set criterion and optimizer
    criterion = torch.nn.BCELoss().to('cuda')
    criterion.requires_grad = True

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # train and eval
    print('#start training#\n')
    for epoch in range(args.epochs):
        train_epoch(args, train_dataloader, model, criterion, optimizer, None)
