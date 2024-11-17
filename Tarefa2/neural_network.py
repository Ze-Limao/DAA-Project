import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from colorama import Fore, init
import matplotlib.pyplot as plt

init(autoreset=True)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class SimpleMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

def initialize_weights(model):
    if isinstance(model, nn.Linear):
        nn.init.kaiming_uniform_(model.weight, nonlinearity='relu')
        if model.bias is not None:
            nn.init.zeros_(model.bias)

def train_model(X_train, y_train, num_epochs=50, learning_rate=0.001):
    input_size = X_train.shape[1]
    num_classes = len(torch.unique(y_train))

    train_size = int(0.8 * len(X_train))
    val_size = len(X_train) - train_size
    train_dataset, val_dataset = random_split(TensorDataset(X_train, y_train), [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = SimpleMLP(input_size, num_classes)
    model.apply(initialize_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Training Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    loss_values = []
    val_loss_values = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        loss_values.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_loss_values.append(avg_val_loss)
        
        scheduler.step()

        ax.clear()
        ax.plot(loss_values, label="Training Loss")
        ax.plot(val_loss_values, label="Validation Loss")
        ax.legend()
        ax.set_title("Training Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        plt.draw()
        plt.pause(0.01)

        print(Fore.YELLOW + f" Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

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