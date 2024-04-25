import torch
from torch.utils.data import Dataset
import csv
import ast
import argparse
from tqdm import tqdm

from utils.model import ScoreModel, FusionScoreModel
from utils.dataset import InferenceDataset, NormalizeTransform, MinMaxNormalizeTransform
from utils.engine import train_epoch, eval_epoch
from utils.logger import Logger

def inference(model, dataloader, output_file, bs):

    scores = []
    with torch.no_grad(), tqdm(total=len(dataloader)*bs) as pbar:
        for emb1, emb2 in dataloader:
            emb1 = emb1.cuda()
            emb2 = emb2.cuda()
            score = model(emb1, emb2).squeeze() 
            scores.extend(score.cpu().numpy().tolist()) 
            pbar.update(bs)

    with open(output_file, 'w') as file:
        for score in scores:
            file.write(f"{score}\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # path and dir
    parser.add_argument("--model_save_path", type=str, 
                        default='/root/Data_Scoring_Model/checkpoints/04%H46.pth')
    # device
    parser.add_argument("--gpu_id", type=str, default='cuda',
                        help="GPU id to work on, \'cpu\'.")
    # data
    parser.add_argument("--batch_size", type=int,
                        default=64, help="batch size of data")      
    parser.add_argument("--num_workers", type=int,
                        default=16, help="number of workers")   
    parser.add_argument("--jsonl_path", type=str,
                        default='/mnt/share_disk/LIV/datacomp/processed_data/1088w_emb/1088w_emb_stats.jsonl', help="path of pairwise dataset csv")   
    parser.add_argument("--res_path", type=str,
                        default='/root/Data_Scoring_Model/res/1088w_score_fusion.txt', help="path of pairwise dataset csv")   
    args = parser.parse_args()

    # get data
    dataset = InferenceDataset(args.jsonl_path,transform=NormalizeTransform(0,1))
    print(len(dataset), 'data ready to go')
    DataLoader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=args.num_workers)
    
    # set model
    model = FusionScoreModel().cuda()
    model.load_state_dict(torch.load(args.model_save_path)['model_state_dict'])
    model.eval()
    print('model ready to go')

    # do inference
    inference(model, DataLoader, args.res_path, args.batch_size)
