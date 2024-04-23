import torch
from PIL import Image
import numpy as np
import argparse
import os
from tqdm import tqdm
import time
import torch.nn.functional as F


def train_epoch(args, dataloader, model, criterion, optimizer, lr_scheduler, logger):

    total_loss = []
    model.train()

    with tqdm(total=len(dataloader)) as pbar: 
       for i, (emb1, emb2, label) in enumerate(dataloader):
            # load to device
            emb1 = emb1.cuda(non_blocking=True)
            emb2 = emb2.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            optimizer.zero_grad()

            with torch.no_grad():
                score1 = model(emb1)
                score2 = model(emb1)

            pred = torch.sigmoid(score1 - score2) 

            # compute loss
            loss = criterion(label, pred)
            # loss backward
            loss.backward()
            # optimizer and scheduler step
            optimizer.step()
            # sum losses in an epoch
            total_loss.append(loss.detach().cpu())
            
            pbar.set_description('training')
            pbar.set_postfix({'loss(iter)':float(loss.detach().cpu()), 
                              'loss(mean)':np.mean(total_loss),
                              'lr':lr_scheduler.get_last_lr()})
            pbar.update(1)

    epoch_loss = np.mean(total_loss)
    #print(f'train_epoch_loss: {epoch_loss}')  
    logger.append(f'train_epoch_loss: {epoch_loss}')
    logger.append(f'train_lr : {lr_scheduler.get_last_lr()[0]}')
     

def evaluate(args, dataloader, model, criterion, optimizer, logger):

    total_loss = []
    model.eval()
    
    with tqdm(total=len(dataloader)) as pbar:
       for i, (emb1, emb2, label) in enumerate(dataloader):
            # load to device
            emb1 = emb1.cuda(non_blocking=True)
            emb2 = emb2.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            optimizer.zero_grad()

            with torch.no_grad():
                score1 = model(emb1)
                score2 = model(emb1)

            pred = torch.sigmoid(score1 - score2) 
            # compute loss
            loss = criterion(label, pred)
            # sum losses in a epoch
            total_loss.append(loss.detach().cpu())
            pbar.set_description('evaluation')
            pbar.set_postfix({'loss(iter)':float(loss.detach().cpu()), 'loss(mean)':np.mean(total_loss)})
            pbar.update(1)

    epoch_loss = np.nanmean(total_loss)
    #print(f'evalutation_epoch_loss: {epoch_loss}') 
    logger.append(f'evalutation_epoch_loss: {epoch_loss}')
