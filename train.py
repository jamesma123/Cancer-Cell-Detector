import numpy as np 
import torch 
import torch.nn as nn
import torch.optim as optim 
from model import SimpleNet
from load_data import train_test_loader

torch.manual_seed(10)


def train(model, train_loader, optimizer, loss_func, epoch, log = 500):
    model.train()
    for batch_idx, (X,y) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_func(output.view(-1), y.view(-1))
        loss.backward()
        optimizer.step()
        if batch_idx % log == 0:
            print("Loss:", loss.item())

def evaluate(model, test_loader):
    model.eval()
    predictions = []
    targets = []
    for batch_idx, (X,y) in enumerate(test_loader):
        output = model(X)
        predictions.append(output.view(-1).cpu().detach().numpy())
        targets.append(y)
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    predictions = (predictions > 0.5).astype(int)
    accuracy = (predictions == targets).sum()/len(predictions)
    return accuracy

lr = 0.01
train_loader, test_loader = train_test_loader()

model = SimpleNet(input_dim=30, output_dim=1)
loss = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(200):
    train(model, train_loader, optimizer, loss, epoch)

print(evaluate(model, test_loader ))
        