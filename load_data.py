import torch
import pandas as pd 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def create_dataloader(dataset, batch_size):
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataLoader

def train_test_loader():
    dataset = load_breast_cancer()
    data = pd.DataFrame(data=dataset['data'], columns=dataset['feature_names'])
    data['target'] = dataset['target']
    y = data['target'].values
    X = data.drop(columns=['target']).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    train_dataset = TensorDataset(torch.Tensor(X_train.astype(float)), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test.astype(float)), torch.Tensor(y_test))
    
    train_loader = create_dataloader(train_dataset, batch_size=50)
    test_loader = create_dataloader(test_dataset, batch_size=50)
    
    return train_loader, test_loader
