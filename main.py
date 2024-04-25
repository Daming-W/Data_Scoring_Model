import os
import datetime
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.multiprocessing as mp

from utils.model import ScoreModel, FusionScoreModel
from utils.dataset import PairDataset, NormalizeTransform, MinMaxNormalizeTransform
from utils.engine import train_epoch, eval_epoch
from utils.logger import Logger

def generate_default_logger_name():
    now = datetime.datetime.now()
    return now.strftime('%m%_%H%M')

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    # path and dir
    parser.add_argument("--model_save_path", type=str, 
                        default='/root/Data_Scoring_Model/checkpoints/'+generate_default_logger_name()+'.pth')
    parser.add_argument("--logger_name", type=str, 
                        default='/root/Data_Scoring_Model/logger/log_'+generate_default_logger_name())

    # device
    parser.add_argument("--gpu_id", type=str, default='cuda',
                        help="GPU id to work on, \'cpu\'.")

    # data
    parser.add_argument("--train_val_ratio", type=float,
                        default=0.8, help="train set and val set dataset ratio")
    parser.add_argument("--batch_size", type=int,
                        default=128, help="batch size of data")      
    parser.add_argument("--num_workers", type=int,
                        default=32, help="number of workers")   
    parser.add_argument("--csv_path", type=str,
                        default='/root/Data_Scoring_Model/data/pair15w.csv', help="path of pairwise dataset csv")   

    # hyperparameters
    parser.add_argument("--lr", type=float,
                        default=1e-7, help="learning rate")   
    parser.add_argument("--weight_decay", type=float,
                        default=0.001, help="weight_decay")
    parser.add_argument("--eps", type=float,
                        default=1e-8, help="eps")
    parser.add_argument("--warmup_ratio", type=float, 
                        default=0.1, help="accumuate gradients")
    parser.add_argument("--epochs", type=int, 
                        default=5, help="total number of training epochs")
    parser.add_argument("--resume", type=bool, 
                        default=False, help="flag to resume training")

    args = parser.parse_args()


    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("Using GPU: ", torch.cuda.get_device_name(0))
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    #load data
    train_dataset = PairDataset(csv_file=args.csv_path, train=True, split_ratio=args.train_val_ratio, transform=NormalizeTransform(0, 1))
    val_dataset = PairDataset(csv_file=args.csv_path, train=False, split_ratio=args.train_val_ratio, transform=NormalizeTransform(0, 1))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print(f'finish loading data train{len(train_dataset)} val{len(val_dataset)}')

    # get model
    model = FusionScoreModel()

    # model to device
    model = model.cuda()

    # setup logger
    logger = Logger(args.logger_name,False)
    logger.append(args)
    print('finish setting logger')

    # set criterion and optimizer
    criterion = torch.nn.BCELoss().to('cuda')
    criterion.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # train and eval
    print('<<< start training >>>')
    for epoch in range(args.epochs):
        print(f'<<< epoch {epoch+1} >>>')
        logger.append(f'epoch : {epoch+1}')
        train_epoch(args, train_dataloader, model, criterion, optimizer, logger)
        eval_epoch(args, val_dataloader, model, criterion, logger)
    
    if args.model_save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, args.model_save_path)
