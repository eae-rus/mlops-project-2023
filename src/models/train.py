import os
os.chdir('../..')

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam

print(os.getcwdb())

from src.data.dataloader import OscillogramDataLoader
from src.models.mlp import MLP


def train():
    # Data preparation for DataLoader:
    df = pd.read_csv('data/interim/data.csv', index_col=['filename', 'Unnamed: 0'])
    train_df = df[df['bus'] == 1]
    test_df = df[df['bus'] == 2]
    
    train_dl = OscillogramDataLoader(
        data=train_df,
        window_size=32,
        step_size=1,
        minibatch_training=True,
        batch_size=256,
        shuffle=True
    )
    
    test_dl = OscillogramDataLoader(
        data=test_df,
        window_size=32,
        step_size=1,
        minibatch_training=True,
        batch_size=256,
        shuffle=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(32)
    model.to(device)
    optimizer = Adam(model.parameters())
    n_epochs = 10
    loss_func = nn.BCELoss()
    
    for e in range(n_epochs):
        model.train()
        av_loss = []
        for train_ts, train_index, train_label in train_dl:
            x = torch.FloatTensor(train_ts)
            y = torch.FloatTensor(train_label)
            logits = model(x)
            pred = torch.squeeze(logits)
            loss = loss_func(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            av_loss.append(loss.item())
        print(f'Epoch: {e+1:2d}/{n_epochs}, average CE loss: {sum(av_loss)/len(av_loss):.4f}')
        

if __name__ == "__main__":
    train()