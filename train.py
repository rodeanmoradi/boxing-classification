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

# Convert from np arrays to PyTorch tensors, split into training and val
training_samples = torch.from_numpy(samples[:164]).float().squeeze(1)
val_samples = torch.from_numpy(samples[164:]).float().squeeze(1)
training_labels = torch.from_numpy(labels[:164]).long()
val_labels = torch.from_numpy(labels[164:]).long()

training_loader = DataLoader(TensorDataset(training_samples, training_labels), 16, shuffle=True)
val_loader = DataLoader(TensorDataset(val_samples, val_labels), 16, shuffle=False)

model = LSTM(34, 32, 1, 2)

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

highest_accuracy = 0
for epoch in range(0, 31):
    model.train()
    correct_training_guesses = 0
    num_training_samples = 164
    for X_training, y_training in training_loader:
        # Forward pass
        output = model(X_training)
        # Calculate loss
        loss = loss_function(output, y_training)
        # Zero gradients
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        # Convert logits to classification
        classification = torch.argmax(output, dim=1)
        # Compare all classifications to y at once
        correct_training_guesses += torch.sum(classification == y_training).item() 
    training_accuracy = correct_training_guesses / num_training_samples
    print(f'Epoch: {epoch}, Training Accuracy: {training_accuracy}')

    model.eval()
    correct_val_guesses = 0
    num_val_samples = 41
    with torch.no_grad():
        for X_val, y_val in val_loader:
            output = model(X_val)
            # Calculate accuracy
            classification = torch.argmax(output, dim=1)
            correct_val_guesses += torch.sum(classification == y_val).item()
    val_accuracy = correct_val_guesses / num_val_samples

    print(f'Epoch: {epoch}, Validation Accuracy: {val_accuracy}')

    if val_accuracy > highest_accuracy:
        highest_accuracy = val_accuracy
        torch.save(model.state_dict(), 'models/LSTM_model.pth')
