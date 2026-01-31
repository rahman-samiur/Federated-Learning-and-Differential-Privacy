import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import csv
import time
from collections import OrderedDict

# --- 1. Model Definition (Same as before) ---


class SimpleCNN(nn.Module):
    """
    A simple CNN model for Fashion-MNIST[cite: 57].
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. Data Loading (IID) ---


def get_dataloaders(num_clients=20, iid=True, batch_size=64):
    """
    Loads Fashion-MNIST [cite: 56] and distributes it[cite: 65].
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )

    test_loader = DataLoader(
        datasets.FashionMNIST(root='./data', train=False, transform=transform),
        batch_size=1024, shuffle=False
    )

    client_dataloaders = []

    if iid:
        # IID: Shuffle and split data equally [cite: 65]
        num_samples_per_client = len(train_dataset) // num_clients
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)

        for i in range(num_clients):
            client_indices = indices[i *
                                     num_samples_per_client: (i + 1) * num_samples_per_client]
            client_subset = Subset(train_dataset, client_indices)
            loader = DataLoader(
                client_subset, batch_size=batch_size, shuffle=True)
            client_dataloaders.append(loader)

    else:
        # Non-IID logic would go here [cite: 66]
        # This baseline will just use IID per the first run.
        print("Using IID data distribution.")
        # (Using IID logic from above)
        num_samples_per_client = len(train_dataset) // num_clients
        indices = list(range(len(train_dataset)))
        np.random.shuffle(indices)

        for i in range(num_clients):
            client_indices = indices[i *
                                     num_samples_per_client: (i + 1) * num_samples_per_client]
            client_subset = Subset(train_dataset, client_indices)
            loader = DataLoader(
                client_subset, batch_size=batch_size, shuffle=True)
            client_dataloaders.append(loader)

    return client_dataloaders, test_loader

# --- 3. Client Update Function (Baseline) ---


def client_update(client_id, model, dataloader, epochs, round_num, csv_writer):
    """
    Implements the Baseline FedAvg client-side training.
    - Trains locally for E epochs [cite: 32]
    - Logs local loss/accuracy to the CSV writer
    - Returns the *un-noised model delta* [cite: 32]
    """
    # Create a local copy of the global model
    local_model = copy.deepcopy(model)
    local_model.train()

    # Store the initial global weights for delta calculation
    global_weights = copy.deepcopy(local_model.state_dict())

    optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for e in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for data, labels in dataloader:
            optimizer.zero_grad()
            outputs = local_model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate and log local training metrics for this epoch
        avg_epoch_loss = epoch_loss / total
        avg_epoch_acc = 100.0 * correct / total

        # Log to CSV
        log_row = [round_num, client_id,
                   f'train_epoch_{e+1}', avg_epoch_loss, avg_epoch_acc]
        csv_writer.writerow(log_row)

        # print(f"  Client {client_id} | Epoch {e+1}/{epochs} | Loss: {avg_epoch_loss:.4f} | Acc: {avg_epoch_acc:.2f}%")

    # After all local epochs, calculate the model delta [cite: 32]
    local_weights = local_model.state_dict()
    model_delta = OrderedDict()
    for key in global_weights:
        model_delta[key] = local_weights[key] - global_weights[key]

    return model_delta

# --- 4. Server Aggregation ---


def server_aggregate(global_model, client_deltas):
    """
    Averages the client deltas and updates the global model[cite: 33].
    """
    # Create a zero-initialized state_dict for the average delta
    avg_delta = OrderedDict()
    for key in client_deltas[0]:
        avg_delta[key] = torch.zeros_like(client_deltas[0][key])

    # Sum all deltas
    for delta in client_deltas:
        for key in delta:
            avg_delta[key] += delta[key]

    # Average the deltas
    for key in avg_delta:
        avg_delta[key] = avg_delta[key] / len(client_deltas)

    # Apply the average delta to the global model
    updated_weights = global_model.state_dict()
    for key in updated_weights:
        updated_weights[key] += avg_delta[key]

    global_model.load_state_dict(updated_weights)

# --- 5. Global Model Evaluation ---


def test_model(model, test_loader, round_num, csv_writer):
    """
    Evaluates the global model on the hold-out test set[cite: 76].
    Logs global loss/accuracy to the CSV writer.
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / total
    avg_acc = 100.0 * correct / total

    print(
        f"**GLOBAL TEST** (Round {round_num}) | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.2f}% **")

    # Log global test metrics to CSV
    log_row = [round_num, 'global', 'test', avg_loss, avg_acc]
    csv_writer.writerow(log_row)

    return avg_acc

# --- 6. Main Training Loop ---


def run_baseline_experiment():
    """
    Orchestrates the entire Baseline FedAvg experiment.
    """

    NUM_CLIENTS = 20     
    NUM_ROUNDS = 5         
    LOCAL_EPOCHS = 2   
    BATCH_SIZE = 64      
    CLIENT_FRACTION = 0.5   
 

    LOG_FILE = 'baseline_fedavg_log.csv'

    # Setup
    print("Setting up baseline experiment...")
    global_model = SimpleCNN()
    client_dataloaders, test_loader = get_dataloaders(
        num_clients=NUM_CLIENTS, iid=False, batch_size=BATCH_SIZE
    )

    num_selected_clients = int(NUM_CLIENTS * CLIENT_FRACTION)

    # Open CSV file for logging
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write the header row
        writer.writerow(['round', 'client_id', 'type', 'loss', 'accuracy'])

        print(f"Starting Baseline FedAvg training for {NUM_ROUNDS} rounds...")

        # Initial test before training
        test_model(global_model, test_loader, 0, writer)

        # --- Federation Loop ---
        for r in range(1, NUM_ROUNDS + 1):
            start_time = time.time()
            print(f"\n--- Round {r}/{NUM_ROUNDS} ---")

            # Select clients [cite: 63]
            selected_client_ids = np.random.choice(
                range(NUM_CLIENTS), num_selected_clients, replace=False
            )
            print(f"Selected clients: {selected_client_ids}")

            client_deltas = []

            # Client training phase
            for client_id in selected_client_ids:
                dataloader = client_dataloaders[client_id]

                # Perform client update and get delta [cite: 32]
                delta = client_update(
                    client_id, global_model, dataloader,
                    LOCAL_EPOCHS, r, writer
                )
                client_deltas.append(delta)

            # Server aggregation phase [cite: 33]
            server_aggregate(global_model, client_deltas)

            # Global model evaluation phase
            test_model(global_model, test_loader, r, writer)

            print(f"Round {r} completed in {time.time() - start_time:.2f}s")

    print(f"\nBaseline training complete. Log saved to {LOG_FILE}")


# --- Run the experiment ---
if __name__ == "__main__":
    run_baseline_experiment()
