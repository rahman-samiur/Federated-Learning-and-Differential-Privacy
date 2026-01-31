import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import json
import time
import os
from collections import OrderedDict

# --- Opacus Imports ---
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

# --- 1. Model Definition (LeNet for CIFAR-10) ---

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- 2. Data Loading (CIFAR-10 Non-IID) ---

def get_non_iid_cifar10_dataloaders(num_clients=100, batch_size=32, alpha=0.5):
    """
    Loads CIFAR-10 and distributes it among clients using Dirichlet partitioning.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    test_loader = DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transform),
        batch_size=1024, shuffle=False
    )

    client_dataloaders = []
    
    # Dirichlet partitioning
    min_size = 0
    min_require_size = 10
    K = 10
    N = len(train_dataset)
    y_train = np.array(train_dataset.targets)
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (proportions * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, np.cumsum(proportions)))]
        min_size = min([len(idx_j) for idx_j in idx_batch])

    for i in range(num_clients):
        client_indices = idx_batch[i]
        client_subset = Subset(train_dataset, client_indices)
        loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True)
        client_dataloaders.append(loader)

    return client_dataloaders, test_loader

# --- 3. Client Update Function (DP-SGD) ---

def dp_sgd_client_update(client_id, model, dataloader, epochs, round_num, 
                         noise_multiplier, max_grad_norm, lr):
    """
    Implements the DP-SGD client-side training.
    """
    local_model = copy.deepcopy(model)
    local_model.train()

    global_weights = copy.deepcopy(local_model.state_dict())

    # --- Opacus Setup ---
    local_model = ModuleValidator.fix(local_model)
    optimizer = optim.Adam(local_model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()

    local_model, optimizer, dataloader = privacy_engine.make_private(
        module=local_model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    # --- End Opacus Setup ---

    for e in range(epochs):
        for data, labels in dataloader:
            optimizer.zero_grad()
            outputs = local_model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Save client weights
    save_dir = f"client_weights_non_iid/param_nm_{noise_multiplier}_cn_{max_grad_norm}"
    os.makedirs(save_dir, exist_ok=True)
    # Partition ID is essentially client_id here
    save_path = os.path.join(save_dir, f"{round_num}_client_{client_id}_weights.pt")
    
    # Save the original model state dict (unwrap Opacus)
    torch.save(local_model._module.state_dict(), save_path)

    local_weights = local_model._module.state_dict()
    model_delta = OrderedDict()
    for key in global_weights:
        model_delta[key] = local_weights[key] - global_weights[key]

    target_delta = 1e-5
    epsilon = privacy_engine.get_epsilon(target_delta)

    return model_delta, epsilon, target_delta

# --- 4. Server Aggregation ---

def server_aggregate(global_model, client_deltas):
    """
    Averages the client deltas and updates the global model.
    """
    avg_delta = OrderedDict()
    for key in client_deltas[0]:
        avg_delta[key] = torch.zeros_like(client_deltas[0][key])

    for delta in client_deltas:
        for key in delta:
            avg_delta[key] += delta[key]

    for key in avg_delta:
        avg_delta[key] = avg_delta[key] / len(client_deltas)

    updated_weights = global_model.state_dict()
    for key in updated_weights:
        updated_weights[key] += avg_delta[key]

    global_model.load_state_dict(updated_weights)

# --- 5. Global Model Evaluation ---

def test_model(model, test_loader, round_num, run_id, log_file, epsilon=0.0, delta=0.0):
    """
    Evaluates the global model and logs to JSONL.
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / total
    avg_acc = correct / total 
    
    print(f"**GLOBAL TEST** (Round {round_num}) | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")

    log_entry = {
        "round": round_num,
        "run_id": run_id,
        "eval_loss": avg_loss,
        "eval_acc": avg_acc,
        "num-examples": total,
        "epsilon": epsilon,
        "delta": delta
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# --- Main Training Loop ---

def run_arm1_experiment(nm, cn):
    """
    Orchestrates the DP-SGD experiment matching flower-test-sv-dp but with Non-IID data.
    """

    # --- Hyperparameters ---
    NUM_CLIENTS = 100
    NUM_ROUNDS = 100
    LOCAL_EPOCHS = 2
    BATCH_SIZE = 32
    CLIENT_FRACTION = 0.5 # 50 clients per round
    LR = 0.001
    ALPHA = 0.5 # Dirichlet alpha

    # --- DP-SGD Specific Parameters ---
    NOISE_MULTIPLIER = nm
    CLIPPING_NORM = cn

    # Filenames
    nm_str = str(NOISE_MULTIPLIER).replace('.', 'p')
    cn_str = str(CLIPPING_NORM).replace('.', 'p')
    LOG_FILE = f"evaluation_logsgd_non_iid_nm{nm_str}_cn{cn_str}.jsonl"
    FINAL_MODEL_FILE = f"final_modelsgd_non_iid_nm{nm_str}_cn{cn_str}.pt"
    
    # Clear log file
    with open(LOG_FILE, "w") as f:
        pass

    # Setup
    print(f"Setting up Non-IID DP-SGD experiment (CIFAR-10, LeNet, Alpha={ALPHA})...")
    global_model = Net()
    client_dataloaders, test_loader = get_non_iid_cifar10_dataloaders(
        num_clients=NUM_CLIENTS, batch_size=BATCH_SIZE, alpha=ALPHA
    )

    num_selected_clients = int(NUM_CLIENTS * CLIENT_FRACTION)
    run_id = str(int(time.time())) 

    print(f"Starting DP-SGD training (NM={NOISE_MULTIPLIER}, CN={CLIPPING_NORM})...")

    # Initial test
    test_model(global_model, test_loader, 0, run_id, LOG_FILE, epsilon=0.0, delta=0.0)

    # --- Federation Loop ---
    for r in range(1, NUM_ROUNDS + 1):
        start_time = time.time()
        print(f"\n--- Round {r}/{NUM_ROUNDS} ---")

        selected_client_ids = np.random.choice(
            range(NUM_CLIENTS), num_selected_clients, replace=False
        )
        
        client_deltas = []
        current_epsilon = 0.0
        current_delta = 0.0

        # Client training phase
        for client_id in selected_client_ids:
            dataloader = client_dataloaders[client_id]

            delta_weights, eps, dlt = dp_sgd_client_update(
                client_id, global_model, dataloader,
                LOCAL_EPOCHS, r,
                NOISE_MULTIPLIER, CLIPPING_NORM, LR
            )
            
            current_epsilon = eps
            current_delta = dlt

            if delta_weights is not None:
                client_deltas.append(delta_weights)

        # Server aggregation phase
        if client_deltas:
            server_aggregate(global_model, client_deltas)

        # Global model evaluation phase
        test_model(global_model, test_loader, r, run_id, LOG_FILE, current_epsilon, current_delta)

        print(f"Round {r} completed in {time.time() - start_time:.2f}s")

    print(f"\nDP-SGD training complete. Log saved to {LOG_FILE}")
    
    # Save final model
    print(f"Saving final model to {FINAL_MODEL_FILE}...")
    torch.save(global_model.state_dict(), FINAL_MODEL_FILE)

if __name__ == "__main__":
    run_arm1_experiment(1.0, 0.1)
    run_arm1_experiment(1.0, 0.5)
    run_arm1_experiment(1.0, 1.0)
    run_arm1_experiment(2.0, 0.1)
    run_arm1_experiment(2.0, 0.5)
    run_arm1_experiment(2.0, 1.0)
    run_arm1_experiment(3.0, 0.1)
    run_arm1_experiment(3.0, 0.5)
    run_arm1_experiment(3.0, 1.0)
