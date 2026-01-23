import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src import LSTM

# Load jab and none sequences into np arrays
jabs = []
nones = []
for num_jabs in range(0, 103):
    jabs.append(np.load(f'data/jab/jab_{num_jabs}.npy'))
jabs = np.array(jabs)
for num_nones in range(0, 102):
    nones.append(np.load(f'data/none/none_{num_nones}.npy'))
nones = np.array(nones)

# Combine arrays and create corresponding labels
samples = np.concatenate((jabs, nones))
labels = np.concatenate((np.ones(103), np.zeros(102)))

# Randomize order
indices = np.random.permutation(205)
samples = samples[indices]
labels = labels[indices]

# Convert from np arrays to PyTorch tensors
training_samples = torch.from_numpy(samples[:164]).float()
val_samples = torch.from_numpy(samples[164:]).float()
training_labels = torch.from_numpy(labels[:164]).long()
val_labels = torch.from_numpy(labels[164:]).long()

training_set = TensorDataset(training_samples, training_labels)
val_set = TensorDataset(val_samples, val_labels)

training_loader = DataLoader(training_set, 16, shuffle=True)
val_loader = DataLoader(val_set, 16, shuffle=False)

model = LSTM(34, 32, 1, 2)

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# TODO: training loop -> forward pass, calculate loss, backward pass, update weights -> print accuracy for each epoch

# validation loop -> print accuracy

# once validation accuracy is best, save model