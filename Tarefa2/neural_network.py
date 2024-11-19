import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from colorama import Fore, init
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import KFold

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
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
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

def train_model(X_train, y_train, num_epochs=50, learning_rate=0.001, patience=5):
    input_size = X_train.shape[1]
    num_classes = len(torch.unique(y_train))

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = SimpleMLP(input_size, num_classes)
    model.apply(initialize_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Training Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    loss_values = []
    val_loss_values = []

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_wts = copy.deepcopy(model.state_dict())

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
            for inputs, labels in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(train_loader)
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(Fore.RED + f"Early stopping at epoch {epoch+1}.")
            break

    plt.ioff()
    plt.show()

    model.load_state_dict(best_model_wts)
    return model

def cross_validate(X_train, y_train, k=5, num_epochs=50, learning_rate=0.001, patience=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    avg_train_loss = 0
    avg_val_loss = 0
    fold = 1
    best_model = None

    for train_index, val_index in kf.split(X_train):
        print(Fore.CYAN + f"Training fold {fold}/{k}...")
        
        train_data, val_data = X_train[train_index], X_train[val_index]
        train_labels, val_labels = y_train[train_index], y_train[val_index]

        model = train_model(train_data, train_labels, num_epochs, learning_rate, patience)
        
        train_loss = 0.0
        val_loss = 0.0
        model.eval()
        
        with torch.no_grad():
            for inputs, labels in DataLoader(TensorDataset(train_data, train_labels), batch_size=32):
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                train_loss += loss.item()

        with torch.no_grad():
            for inputs, labels in DataLoader(TensorDataset(val_data, val_labels), batch_size=32):
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                val_loss += loss.item()

        avg_train_loss += train_loss / len(train_data)
        avg_val_loss += val_loss / len(val_data)
        
        print(Fore.GREEN + f"Fold {fold} - Train Loss: {train_loss / len(train_data):.4f}, Val Loss: {val_loss / len(val_data):.4f}")
        
        if best_model is None or val_loss / len(val_data) < avg_val_loss / k:
            best_model = model

        fold += 1

    print(Fore.MAGENTA + f"Average Training Loss: {avg_train_loss / k:.4f}, Average Validation Loss: {avg_val_loss / k:.4f}")
    
    return best_model

def main(X_train, y_train):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    best_model = cross_validate(X_train, y_train, k=5, num_epochs=50)
    print(Fore.GREEN + "✅ Cross-validation completed.")
    return best_model

if __name__ == "__main__":
    pass