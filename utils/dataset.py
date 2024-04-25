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
            selected_raw_emb_img = [em['__dj__stats__']['image_embedding'][0] for em in selected_raw_emb]
            selected_raw_emb_txt = [em['__dj__stats__']['text_embedding'][0] for em in selected_raw_emb]

            selected_eval_emb = random.sample(all_eval_emb,size)
            selected_eval_emb_img = [em['__dj__stats__']['image_embedding'][0] for em in selected_eval_emb]
            selected_eval_emb_txt = [em['__dj__stats__']['text_embedding'][0] for em in selected_eval_emb]

            print('random getting samples')
            cnt=0
            with tqdm(total=len(selected_raw_emb), desc="Writing to CSV") as pbar:  
                for e1,e2,e3,e4 in zip(selected_raw_emb_img,selected_raw_emb_txt,selected_eval_emb_img,selected_eval_emb_txt):
                    # set label=0 -> [dirty, clean, 0]

                    if cnt<=size//2:
                        writer.writerow([e1,e2,e3,e4,0])  
                        pbar.update()
                    # set label=1 -> [clean, dirty, 1]
                    else:    
                        writer.writerow([e3,e4,e1,e2,1])  
                        pbar.update()
                    cnt+=1
    return None


class NormalizeTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, embeddings):
        return [(x - self.mean) / self.std for x in embeddings]

class MinMaxNormalizeTransform:
    def __init__(self, feature_min, feature_max):
        self.feature_min = feature_min
        self.feature_max = feature_max

    def __call__(self, embeddings):
        normalized = [(x - self.feature_min) / (self.feature_max - self.feature_min) for x in embeddings]
        return normalized
    
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
        emb3 = ast.literal_eval(self.data[idx][2])
        emb4 = ast.literal_eval(self.data[idx][3])
        label = int(self.data[idx][4])

        if self.transform:
            emb1 = self.transform(emb1)
            emb2 = self.transform(emb2)
            emb3 = self.transform(emb3)
            emb4 = self.transform(emb4)
        
        emb1_tensor = torch.tensor(emb1, dtype=torch.float32)
        emb2_tensor = torch.tensor(emb2, dtype=torch.float32)
        emb3_tensor = torch.tensor(emb3, dtype=torch.float32)
        emb4_tensor = torch.tensor(emb4, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return emb1_tensor,emb2_tensor,emb3_tensor,emb4_tensor,label_tensor
    
class InferenceDataset(Dataset):

    def __init__(self, jsonl_path,  transform=None):
        with jsonlines.open(jsonl_path,'r') as raw:
            self.all_raw_emb = list(raw)

            self.img_emb = [em['__dj__stats__']['image_embedding'][0] for em in self.all_raw_emb]
            self.txt_emb = [em['__dj__stats__']['text_embedding'][0] for em in self.all_raw_emb]
            self.sim_score = [em['__dj__stats__']['image_text_similarity'][0] for em in self.all_raw_emb]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        emb1 = self.img_emb[idx][0]
        emb2 = self.txt_emb[idx][1]
        score = float(self.sim_score[idx][0])

        if self.transform:
            emb1 = self.transform(emb1)
            emb2 = self.transform(emb2)

        emb1_tensor = torch.tensor(emb1, dtype=torch.float32)
        emb2_tensor = torch.tensor(emb2, dtype=torch.float32)

        return emb1_tensor, emb2_tensor


if __name__=='__main__':

    make_dataset(
        '/root/Data_Scoring_Model/data/commonpool_stats_50w.jsonl',
        '/root/Data_Scoring_Model/data/imagenet_emb_stats.jsonl',
        '/root/Data_Scoring_Model/data/pair15w.csv',
        150000)

    # with open('/root/QCM/data/pair100000.csv','r') as csv_file:
    #     csv_reader = csv.reader(csv_file)  
    #     rows = list(csv_reader) 
    #     for row in rows:
    #         if int(row[2])==1:
    #             print(row)