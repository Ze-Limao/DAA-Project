import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from colorama import Fore, init
import matplotlib.pyplot as plt

init(autoreset=True)

class SimpleMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

def train_model(X_train, y_train, num_epochs=50, learning_rate=0.001):
    input_size = X_train.shape[1]
    num_classes = len(torch.unique(y_train))

    model = SimpleMLP(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Training Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    loss_values = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        loss_values.append(avg_loss)
        ax.clear()
        ax.plot(loss_values, label="Loss")
        ax.legend()
        ax.set_title("Training Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        plt.draw()
        plt.pause(0.01)

        print(Fore.YELLOW + f" Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    plt.ioff()
    plt.show()

    return model

def main(X_train, y_train):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    model = train_model(X_train, y_train)
    torch.save(model.state_dict(), "trained_mlp_model.pth")
    print(Fore.GREEN + "✅ Model training completed and saved.")
    return model

if __name__ == "__main__":
    pass
