import torch
from torch.utils.data import Dataset
import csv
import ast
import argparse

from utils.model import ScoreModel
from utils.dataset import InferenceDataset, NormalizeTransform, MinMaxNormalizeTransform
from utils.engine import train_epoch, eval_epoch
from utils.logger import Logger

def load_model(model_path):
    model = ScoreModel().cuda()
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    return model

def inference(model, dataset, output_file, batch_size):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores = []
    with torch.no_grad():
        for emb1, emb2 in loader:
            emb1 = emb1.cuda()
            emb2 = emb2.cuda()
            score = model(emb1, emb2).squeeze()  # Make sure to squeeze if the output has unnecessary dims
            scores.extend(score.cpu().numpy().tolist())  # Store scores as list of floats

    with open(output_file, 'w') as file:
        for score in scores:
            file.write(f"{score}\n")


# 使用示例
if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    # path and dir
    parser.add_argument("--model_save_path", type=str, 
                        default='/root/Data_Scoring_Model/checkpoints/')
    # device
    parser.add_argument("--gpu_id", type=str, default='cuda',
                        help="GPU id to work on, \'cpu\'.")

    # data
    parser.add_argument("--train_val_ratio", type=float,
                        default=0.8, help="train set and val set dataset ratio")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="batch size of data")      
    parser.add_argument("--num_workers", type=int,
                        default=16, help="number of workers")   
    parser.add_argument("--csv_path", type=str,
                        default='/root/Data_Scoring_Model/data/pair15w.csv', help="path of pairwise dataset csv")   
    parser.add_argument("--res_path", type=str,
                        default='/root/Data_Scoring_Model/data/pair15w.csv', help="path of pairwise dataset csv")   


    args = parser.parse_args()

    # get data
    dataset = InferenceDataset(args.csv_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    print(len(dataset), 'data ready to go')

    # set model
    model = load_model("path_to_model.pth")
    print('model ready to go')

    # do inference
    inference(model, dataset, args.res_path)
