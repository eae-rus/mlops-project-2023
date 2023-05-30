import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import f1_score
import mlflow
from mlflow.models.signature import infer_signature
from mlflow import log_metric
import os

from models import MLP
from data import OscillogramDataLoader

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # need to change IP adress
mlflow.set_experiment('mlp')
mlflow.pytorch.autolog()
with mlflow.start_run():
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
    )

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = MLP(32)
    model.to(device)
    optimizer = Adam(model.parameters())
    n_epochs = 10
    loss_func = nn.BCELoss()

    print('\n Training:')

    model.train()
    for e in range(n_epochs):
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

    print('\n Testing:')
    model.eval()
    preds = []
    test_labels = []
    for test_ts, test_index, test_label in test_dl:
        x = torch.FloatTensor(test_ts)
        with torch.no_grad():
            logits = model(x)
        p = torch.round(torch.squeeze(logits))
        preds.append(pd.Series(p, index=test_index))
        test_labels.append(test_label)
    pred = pd.concat(preds)
    test_label = pd.concat(test_labels)
    score = f1_score(test_label, pred)
    print('f1_score: ', score)
    log_metric("f1", score)

    signature = infer_signature(test_df, pred)
    mlflow.pytorch.log_model(model, "signals", signature=signature)

    # autolog_run = mlflow.last_active_run()

    # torch.save(model, 'models/model.pt')


# if __name__ == "__main__":
#     train()
