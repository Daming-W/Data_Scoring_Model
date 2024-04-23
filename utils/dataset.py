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

# generate dataset for pairwise learning
def make_dataset(raw_jsonl_path, eval_jsonl_path, dataset_csv_path, size):
    size0 = size1 = size//2

    with jsonlines.open(raw_jsonl_path,'r') as raw,jsonlines.open(eval_jsonl_path,'r') as eval:
        # get all samples
        with open(dataset_csv_path, mode='w', newline='') as csv_file:  
            writer = csv.writer(csv_file)  

            all_raw_emb = list(raw)
            print('finish loading raw')
            all_eval_emb = list(eval)
            print('finish loading eval')

            selected_raw_emb = random.sample(all_raw_emb['__dj__stats__']['image_embedding'][0],size//2)
            selected_eval_emb = random.sample(all_eval_emb['__dj__stats__']['image_embedding'][0],size//2)
            print('random getting samples')

            label = [1]*(size//2)
            with tqdm(total=len(selected_raw_emb), desc="Writing to CSV") as pbar:  
                for e1,e2,l in zip(selected_raw_emb,selected_eval_emb,label):
                    writer.writerow([e1,e2,l])  
                    pbar.update()
        
    return None


class pairw(Dataset):

    def __init__(self,args,path):
        self.args = args
        self.path = path


         
    def __len__(self):
        data_size = len(self.csv_file)
        return data_size
    
if __name__=='__main__':

    make_dataset(
        '/mnt/share_disk/LIV/datacomp/processed_data/1088w_emb/1088w_emb_stats.jsonl',
        '/mnt/share_disk/LIV/datacomp/processed_data/evalset_emb/evalset_emb_stats.jsonl',
        '/root/QCM/data/pair5000.csv',
        5000)