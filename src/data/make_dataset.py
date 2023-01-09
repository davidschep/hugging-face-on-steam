# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


class CorruptMnist(Dataset):
    def __init__(self, in_folder: str = "", out_folder: str = "") -> None:
        super().__init__()

        self.in_folder = in_folder
        self.out_folder = out_folder
        
        # TODO: refactor out path
        path = str(Path(__file__).parent)
        print("PATH: " + path)
        self.train_np_loader, self.test_np_loader = self.load_data(path)
            
    def load_data(this, path):
        # exchange with the corrupted mnist dataset
        #train = torch.randn(50000, 784)
        #test = torch.randn(10000, 784) 
        
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
        trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
        
        train_np = np.load(path + "\\corruptmnist\\train_0.npz")
        test_np = np.load(path + "\\corruptmnist\\test.npz")
        train = list(zip(train_np['images'], train_np['labels']))
        test = list(zip(test_np['images'], test_np['labels']))
        train_np_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
        test_np_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)
        
        return train_np_loader, test_np_loader
    

class SteamKaggleReviews(Dataset):
    def __init__(self, in_folder: str = "", out_folder: str = "") -> None:
        print("Implement..")


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath: str, output_filepath: str) -> None:
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    train = CorruptMnist(in_folder=input_filepath, out_folder=output_filepath)
    #test = CorruptMnist(train=False, in_folder=input_filepath, out_folder=output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
  