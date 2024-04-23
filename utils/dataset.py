import torch
from PIL import Image
import numpy as np
import argparse
import random
import json
import jsonlines
import csv
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import ast

# generate dataset for pairwise learning
def make_dataset(raw_jsonl_path, eval_jsonl_path, dataset_csv_path, size):

    with jsonlines.open(raw_jsonl_path,'r') as raw,jsonlines.open(eval_jsonl_path,'r') as eval:
        # get all samples
        with open(dataset_csv_path, mode='w', newline='') as csv_file:  
            writer = csv.writer(csv_file)  

            all_raw_emb = list(raw)
            print('finish loading raw')
            all_eval_emb = list(eval)
            print('finish loading eval')

            selected_raw_emb = random.sample(all_raw_emb,size)
            selected_raw_emb = [em['__dj__stats__']['image_embedding'][0] for em in selected_raw_emb]

            selected_eval_emb = random.sample(all_eval_emb,size)
            selected_eval_emb = [em['__dj__stats__']['image_embedding'][0] for em in selected_eval_emb]

            print('random getting samples')
            cnt=0
            with tqdm(total=len(selected_raw_emb), desc="Writing to CSV") as pbar:  
                for e1, e2 in zip(selected_raw_emb, selected_eval_emb):
                    # set label=0 -> [dirty, clean, 0]
                    print(len(e1),len(e2))
                    if cnt<=size//2:
                        writer.writerow([e1,e2,0])  
                        pbar.update()
                    # set label=1 -> [clean, dirty, 1]
                    else:    
                        writer.writerow([e2,e1,1])  
                        pbar.update()
                    cnt+=1
    return None


class PairDataset(Dataset):

    def __init__(self, csv_file, train=True, split_ratio=0.8, transform=None):
        # transform if need
        self.transform = transform
        # load all data points
        data = []
        with open(csv_file, mode='r') as file:
            csv_reader = csv.reader(file)
            data = [row for row in csv_reader]
        # for splitting train/val
        if train:
            self.data = data[:int(len(data) * split_ratio)]
        else:
            self.data = data[int(len(data) * split_ratio):]
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):

        emb1 = ast.literal_eval(self.data[idx][0])
        emb2 = ast.literal_eval(self.data[idx][1])
        label = int(self.data[idx][2])
 
        if self.transform:
            emb1 = self.transform(emb1)
            emb2 = self.transform(emb2)
        
        emb1_tensor = torch.tensor(emb1, dtype=torch.float32).cuda(non_blocking=True)
        emb2_tensor = torch.tensor(emb2, dtype=torch.float32).cuda(non_blocking=True)
        label_tensor = torch.tensor(label, dtype=torch.float32).cuda(non_blocking=True)

        return emb1_tensor, emb2_tensor, label_tensor
    

if __name__=='__main__':

    make_dataset(
        '/root/QCM/data/test_10.jsonl',
        '/mnt/share_disk/LIV/datacomp/processed_data/evalset_emb/evalset_emb_stats.jsonl',
        '/root/QCM/data/pair10.csv',
        10)

    # with open('/root/QCM/data/pair100000.csv','r') as csv_file:
    #     csv_reader = csv.reader(csv_file)  
    #     rows = list(csv_reader) 
    #     for row in rows:
    #         if int(row[2])==1:
    #             print(row)