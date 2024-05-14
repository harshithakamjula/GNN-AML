import datetime
import os
from typing import Callable, Optional
import pandas as pd
from sklearn import preprocessing
import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset
)

pd.set_option('display.max_columns', None)

from google.colab import drive
drive.mount('/content/drive')

"""# Model Training and evaluation

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, Linear
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score
import pandas as pd
from sklearn import preprocessing
import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, int(hidden_channels/4), heads=1, concat=False, dropout=0.6)
        self.lin = Linear(int(hidden_channels/4), out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.lin(x)
        return x

class AMLtoGraph(InMemoryDataset):

    def __init__(self, root: str, edge_window_size: int = 10,
                 transform: T.Compose = None,
                 pre_transform: T.Compose = None):
        self.edge_window_size = edge_window_size
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return 'LI-Medium_Trans.csv' #Name of small/medium transactions file

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_nodes(self) -> int:
        return self._data.edge_index.max().item() + 1

    def df_label_encoder(self, df, columns):
        le = preprocessing.LabelEncoder()
        for i in columns:
            df[i] = le.fit_transform(df[i].astype(str))
        return df

    def preprocess(self, df):
        df = self.df_label_encoder(df,['Payment Format', 'Payment Currency', 'Receiving Currency'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Timestamp'] = df['Timestamp'].apply(lambda x: x.value)
        df['Timestamp'] = (df['Timestamp']-df['Timestamp'].min())/(df['Timestamp'].max()-df['Timestamp'].min())

        df['Account'] = df['From Bank'].astype(str) + '_' + df['Account']
        df['Account.1'] = df['To Bank'].astype(str) + '_' + df['Account.1']
        df = df.sort_values(by=['Account'])
        receiving_df = df[['Account.1', 'Amount Received', 'Receiving Currency']]
        paying_df = df[['Account', 'Amount Paid', 'Payment Currency']]
        receiving_df = receiving_df.rename({'Account.1': 'Account'}, axis=1)
        currency_ls = sorted(df['Receiving Currency'].unique())

        return df, receiving_df, paying_df, currency_ls

    def get_all_account(self, df):
        ldf = df[['Account', 'From Bank']]
        rdf = df[['Account.1', 'To Bank']]
        suspicious = df[df['Is Laundering']==1]
        s1 = suspicious[['Account', 'Is Laundering']]
        s2 = suspicious[['Account.1', 'Is Laundering']]
        s2 = s2.rename({'Account.1': 'Account'}, axis=1)
        suspicious = pd.concat([s1, s2], join='outer')
        suspicious = suspicious.drop_duplicates()

        ldf = ldf.rename({'From Bank': 'Bank'}, axis=1)
        rdf = rdf.rename({'Account.1': 'Account', 'To Bank': 'Bank'}, axis=1)
        df = pd.concat([ldf, rdf], join='outer')
        df = df.drop_duplicates()

        df['Is Laundering'] = 0
        df.set_index('Account', inplace=True)
        df.update(suspicious.set_index('Account'))
        df = df.reset_index()
        return df

    def paid_currency_aggregate(self, currency_ls, paying_df, accounts):
        for i in currency_ls:
            temp = paying_df[paying_df['Payment Currency'] == i]
            accounts['avg paid '+str(i)] = temp['Amount Paid'].groupby(temp['Account']).transform('mean')
        return accounts

    def received_currency_aggregate(self, currency_ls, receiving_df, accounts):
        for i in currency_ls:
            temp = receiving_df[receiving_df['Receiving Currency'] == i]
            accounts['avg received '+str(i)] = temp['Amount Received'].groupby(temp['Account']).transform('mean')
        accounts = accounts.fillna(0)
        return accounts

    def get_edge_df(self, accounts, df):
        accounts = accounts.reset_index(drop=True)
        accounts['ID'] = accounts.index
        mapping_dict = dict(zip(accounts['Account'], accounts['ID']))
        df['From'] = df['Account'].map(mapping_dict)
        df['To'] = df['Account.1'].map(mapping_dict)
        df = df.drop(['Account', 'Account.1', 'From Bank', 'To Bank'], axis=1)

        edge_index = torch.stack([torch.from_numpy(df['From'].values), torch.from_numpy(df['To'].values)], dim=0)

        df = df.drop(['Is Laundering', 'From', 'To'], axis=1)

        edge_attr = torch.from_numpy(df.values).to(torch.float)
        return edge_attr, edge_index

    def get_node_attr(self, currency_ls, paying_df,receiving_df, accounts):
        node_df = self.paid_currency_aggregate(currency_ls, paying_df, accounts)
        node_df = self.received_currency_aggregate(currency_ls, receiving_df, node_df)
        node_label = torch.from_numpy(node_df['Is Laundering'].values).to(torch.float)
        node_df = node_df.drop(['Account', 'Is Laundering'], axis=1)
        node_df = self.df_label_encoder(node_df,['Bank'])
        node_df = torch.from_numpy(node_df.values).to(torch.float)
        return node_df, node_label

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        df, receiving_df, paying_df, currency_ls = self.preprocess(df)
        accounts = self.get_all_account(df)
        node_attr, node_label = self.get_node_attr(currency_ls, paying_df,receiving_df, accounts)
        edge_attr, edge_index = self.get_edge_df(accounts, df)

        data = Data(x=node_attr,
                    edge_index=edge_index,
                    y=node_label,
                    edge_attr=edge_attr
                    )

        data_list = [data]
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(data.x, data.edge_index, data.edge_attr)
            loss = self.criterion(pred.squeeze(), data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch: {epoch}, Loss: {avg_loss:.4f}')
        return avg_loss

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                pred = self.model(data.x, data.edge_index, data.edge_attr)
                pred = (pred > 0.5).int()
                correct += (pred.squeeze() == data.y).sum().item()
                total += len(data.y)
        accuracy = correct / total
        print(f'Accuracy: {accuracy:.4f}')
        return accuracy

    def fit(self, train_loader, test_loader, epochs):
        train_losses = []
        test_accuracies = []
        for epoch in range(1, epochs + 1):
            train_loss = self.train(train_loader, epoch)
            test_acc = self.evaluate(test_loader)
            train_losses.append(train_loss)
            test_accuracies.append(test_acc)
        return train_losses, test_accuracies

dataset = AMLtoGraph('/content/drive/MyDrive/AMLdatasets')
data = dataset[0]

def objective(params, data):
    hidden_channels, heads, lr = params
    model = GAT(
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        out_channels=1,
        heads=heads
    ).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    split = T.RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.1)
    split(data)

    train_loader = NeighborLoader(data, num_neighbors=[30] * 2, batch_size=2048)
    test_loader = NeighborLoader(data, num_neighbors=[30] * 2, batch_size=2048)

    trainer = Trainer(model, criterion, optimizer, device)
    train_losses, test_accuracies = trainer.fit(train_loader, test_loader, epochs=10)

    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.edge_attr)
            pred = (pred > 0.5).int().cpu().numpy()
            true_labels.extend(data.y.cpu().numpy())
            pred_labels.extend(pred)

    f1 = f1_score(true_labels, pred_labels)
    return f1

# Define the hyperparameter grid
hyperparams_grid = {
    'hidden_channels': [16, 32, 64, 128, 256],
    'heads': [1, 2, 4, 8],
    'lr': [1e-5, 1e-4, 1e-3, 1e-2]
}

# Generate all combinations of hyperparameters
hyperparams_combinations = list(itertools.product(*hyperparams_grid.values()))

best_f1_score = 0
best_params = {}

for params in hyperparams_combinations:
    # Train the model with the current hyperparameters
    f1_score_result = objective(params, data)

    # Update the best parameters if the current run achieved a better F1 score
    if f1_score_result > best_f1_score:
        best_f1_score = f1_score_result
        best_params = {'hidden_channels': params[0], 'heads': params[1], 'lr': params[2]}

print("Best F1 Score:", best_f1_score)
print("Best Parameters:", best_params)
