import argparse
import sys

import click
import torch
from model import MyAwesomeModel
from torch import nn, optim

from src.data.make_dataset import CorruptMnist


def train(lr):
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    dataset = CorruptMnist()
    train_set = dataset.test_np_loader
    test_set = dataset.train_np_loader
    
    # Define the loss
    criterion = nn.NLLLoss()
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 3
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
    torch.save(model.state_dict(), "models/trained_model.pt")
        
train(0.0)