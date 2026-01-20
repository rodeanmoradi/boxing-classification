import numpy as np
import torch as pt
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src import LSTM

# load data

jabs = []
for num_jabs in range(0, 103):
    jabs.append(np.load(f'data/jab/jab_{num_jabs}.npy'))

nones = []
for num_nones in range(0, 102):
    nones.append(np.load(f'data/none/none_{num_nones}.npy'))

# Use DataLoader to create batches

# Initialize model -> choose hidden_size and num_layers

# Loss function

# Optimizer to minimze loss

# training loop -> forward pass, calculate loss, backward pass, update weights -> print accuracy for each epoch

# validation loop -> print accuracy

# once validation accuracy is best, save model