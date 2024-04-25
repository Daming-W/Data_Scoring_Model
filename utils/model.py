import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreModel(nn.Module):
    def __init__(self):
        super(ScoreModel, self).__init__()

        self.shared_mlp = nn.Sequential(
            nn.Linear(768*2, 768), 
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Linear(768, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, img_embedding, txt_embedding):
        input = torch.cat((img_embedding, txt_embedding),axis=0)
        score = self.shared_mlp(input)
        return score



class FusionScoreModel(nn.Module):
    def __init__(self):
        super(FusionScoreModel, self).__init__()

        self.embed_dim = 768
        self.token_embed_dim = self.embed_dim * 2  

        self.img_token = nn.Parameter(torch.randn(1, self.embed_dim))
        self.txt_token = nn.Parameter(torch.randn(1, self.embed_dim))

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_embed_dim,
            nhead=8,
            dim_feedforward=self.token_embed_dim * 4,
            dropout=0.1,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=3)

        self.mlp = nn.Sequential(
            nn.Linear(self.token_embed_dim, 768),
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.Dropout(0.2),
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, img_embedding, txt_embedding):

        img_input = torch.cat([self.img_token.repeat(img_embedding.size(0), 1), img_embedding], dim=1)
        txt_input = torch.cat([self.txt_token.repeat(txt_embedding.size(0), 1), txt_embedding], dim=1)


        combined_input = torch.cat([img_input, txt_input], dim=1).unsqueeze(0)  

        transformer_output = self.transformer_encoder(combined_input)
        transformer_output = transformer_output.squeeze(0)

        score = self.mlp(transformer_output)
        return score
