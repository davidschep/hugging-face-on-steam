import argparse
import sys

import torch
import click

from data import mnist
from torch import nn, optim

import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x
    
def train(lr):
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    train_set, test_set = mnist()
    
    # Define the loss
    criterion = nn.NLLLoss()
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 30
    steps = 0

    train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:
            
            optimizer.zero_grad()
            
            log_ps = model(images.float())
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        else:
            tot_test_loss = 0
            test_correct = 0  # Number of correct predictions on the test set
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                for images, labels in test_set:
                    log_ps = model(images.float())
                    loss = criterion(log_ps, labels)
                    tot_test_loss += loss.item()

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    test_correct += equals.sum().item()

            # Get mean loss to enable comparison between train and test sets
            train_loss = running_loss / len(train_set.dataset)
            test_loss = tot_test_loss / len(test_set.dataset)

            # At completion of epoch
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                "Training Loss: {:.3f}.. ".format(train_loss),
                "Test Loss: {:.3f}.. ".format(test_loss),
                "Test Accuracy: {:.3f}".format(test_correct / len(test_set.dataset)))
