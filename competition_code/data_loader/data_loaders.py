from torchvision import datasets, transforms
from base import BaseDataLoader
from six.moves import urllib
from parse_config import ConfigParser

# downloads
import requests
import json
from collections import Counter 
import os
import errno
import csv
import numpy as np
import pandas as pd
import splitfolders
import pathlib    
import torchaudio
import torch


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    #def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, weighted_sample = False, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # headers required for valid request to cloudflare-protected dataset
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


# Note: horizontal dimension = 2 * time_window * sample_rate // n_fft + 1
# vertical crop = n_fft // 2 + 1
class PickledSpectrogramLoader(BaseDataLoader):
    def __init__(self, dataset=None, batch_size=128, shuffle=False, validation_split=0.0, weighted_sample = False, num_workers=1, data_dir="data/processed", training=True):
        self.dataset = dataset        
        self.data_dir = data_dir   
        self.vertical_crop = 128     
        self.horizontal_crop = 281       
        if dataset is not None:
            if dataset.mode == 'xeno':
            # Stack of numpy melspecs -> one torch melspec
            #self.horizontal_crop=dataset.horizontal_crop - 1
                trsfm = transforms.Compose([
                    RandomImage(),
                    #AddChannel(),
                    ThreeChannel(),
                    NumpyStackToTensors()

                    #transforms.ToTensor()
                #transforms.RandomCrop(size = (self.vertical_crop, self.horizontal_crop), pad_if_needed=True, padding_mode = 'constant')
            ])
            else:
                trsfm = transforms.Compose([
                    # RandomImage(),
                    ThreeChannel(),
                    AxisOrderChange(),
                    NumpyStackToTensors()
                    #transforms.ToTensor(),
                    #transforms.RandomCrop(size = (self.vertical_crop, self.horizontal_crop), pad_if_needed=True, padding_mode = 'constant')
                ])
            dataset.set_transform(trsfm)
        else:
            dataset = datasets.DatasetFolder(root = self.data_dir, loader = self.default_loader, transform = trsfm, extensions=('.pickle'))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, weighted_sample, num_workers)


    # assumes we have used torch.save() or another pickle saver
    # on tensor-based spectrogram
    def default_loader(self, path):
        mel_specgram = torch.load(path)
        return mel_specgram.numpy()

class AddChannel(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        new_sample = sample[:, :, None]
        return new_sample

class RandomImage(object):
    # Pick a random image from a stack
    def __call__(self, sample):
        choices = range(sample.shape[0])
        choice = np.random.choice(choices)
        new_sample = sample[choice]
        return new_sample

class ThreeChannel(object):
    # Converts a stack of images to color
    def __call__(self, sample):
        sample = np.stack([sample, sample, sample])
        return sample

class NumpyStackToTensors(object):
    def __call__(self, sample):
        sample = [transforms.ToTensor()(sample[i]) for i in range(sample.shape[0])]
        sample = torch.stack(sample)
        return torch.squeeze(sample)   

class AxisOrderChange(object):
    # Torch tensor transform expects:
    # HxWxC, from 0 to 255
    # Returns CxHxW
    def __call__(self, sample):
        sample = np.moveaxis(sample, 0, -1)
        return sample
                     