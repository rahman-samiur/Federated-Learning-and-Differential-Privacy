import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import csv
import time
from collections import OrderedDict

# --- Opacus import for noise calibration ---
# We use this to fairly calculate the noise sigma from epsilon
from opacus.accountants.utils import get_noise_multiplier

# --- Imports from baseline_fl.py ---
try:
    from baseline_fl import (
        SimpleCNN,
        get_dataloaders,
        server_aggregate,
        test_model
    )
except ImportError:
    print("Error: Could not import from baseline_fl.py.")
    print("Please ensure baseline_fl.py is in the same directory and contains:")
    print("SimpleCNN, get_dataloaders, server_aggregate, and test_model.")
    exit()

# --- Client Update Function (Arm 2: One-Shot Perturbation) ---


def arm2_perturb_client_update(client_id, model, dataloader, epochs, round_num, csv_writer,
                               target_epsilon, l2_sensitivity_S):
    """
    Implements the Arm 2: One-Shot Model Perturbation.
    - Trains locally (non-privately)
    - Clips the *total model delta*
    - Adds one-shot calibrated noise to the delta
    """
    local_model = copy.deepcopy(model)
    local_model.train()

    # Store the initial global weights for delta calculation
    global_weights = copy.deepcopy(local_model.state_dict())

    optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # --- 1. Standard Non-Private Local Training ---
    # (This is the same as the baseline client)
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

        avg_epoch_loss = epoch_loss / total if total > 0 else 0
        avg_epoch_acc = 100.0 * correct / total if total > 0 else 0

        log_row = [round_num, client_id,
                   f'train_epoch_{e+1}', avg_epoch_loss, avg_epoch_acc]
        csv_writer.writerow(log_row)

    # --- 2. Calculate Model Delta ---
    local_weights = local_model.state_dict()
    model_delta = OrderedDict()
    for key in global_weights:
        model_delta[key] = local_weights[key] - global_weights[key]

    # --- 3. Clip the Entire Delta (One-Shot) ---
    # This is the "Total Clipping Method" from [cite: 43]

    # First, compute the total L2 norm of the *entire* delta
    flat_delta = torch.cat([p.view(-1) for p in model_delta.values()])
    total_norm = torch.norm(flat_delta, p=2)

    # Calculate clipping factor per [cite: 50]
    clipping_factor = min(1.0, l2_sensitivity_S / (total_norm.item() + 1e-6))

    # Apply clipping
    if clipping_factor < 1.0:
        for key in model_delta:
            model_delta[key] *= clipping_factor

    # --- 4. Add Calibrated Gaussian Noise ---
    # Get noise multiplier (sigma) from epsilon for a single-step (one-shot) mechanism
    client_dataset_size = len(dataloader.dataset)
    if client_dataset_size == 0:
        print(f"Warning: Client {client_id} has no data. Skipping.")
        return None

    target_delta = 1.0 / client_dataset_size

    # We use sample_rate=1 and steps=1 to model a single-shot mechanism
    # This fairly calculates the sigma needed to achieve target_epsilon [cite: 73]
    sigma_multiplier = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=1.0,
        steps=1
    )

    # Calculate noise standard deviation per [cite: 51]
    # Noise is N(0, (S * sigma)^2 * I)
    noise_std = l2_sensitivity_S * sigma_multiplier

    # Add Gaussian noise to the (now clipped) delta
    for key in model_delta:
        noise = torch.normal(0, noise_std, size=model_delta[key].shape)
        model_delta[key] += noise

    # Return the noisy, clipped delta
    return model_delta

# --- Main Training Loop ---


def run_arm2_experiment():
    """
    Orchestrates the Arm 2: One-Shot Model Perturbation experiment.
    """

    # --- Hyperparameters ---
    NUM_CLIENTS = 20
    NUM_ROUNDS = 5
    LOCAL_EPOCHS = 2
    BATCH_SIZE = 64
    CLIENT_FRACTION = 0.5

    # --- DP-Specific Parameters ---
    TARGET_EPSILON = 50.0
    # This is 'S', the L2 sensitivity (clipping bound) for the *total delta*
    # This is a key hyperparameter you will need to tune.
    L2_SENSITIVITY = 1.0

    LOG_FILE = 'arm2_one_shot_log.csv'

    # Setup
    print("Setting up Arm 2: One-Shot Perturbation experiment...")
    global_model = SimpleCNN()
    client_dataloaders, test_loader = get_dataloaders(
        num_clients=NUM_CLIENTS, iid=True, batch_size=BATCH_SIZE
    )

    num_selected_clients = int(NUM_CLIENTS * CLIENT_FRACTION)

    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'client_id', 'type', 'loss', 'accuracy'])

        print(
            f"Starting One-Shot training (target_epsilon={TARGET_EPSILON}, S={L2_SENSITIVITY})...")

        test_model(global_model, test_loader, 0, writer)

        # --- Federation Loop ---
        for r in range(1, NUM_ROUNDS + 1):
            start_time = time.time()
            print(f"\n--- Round {r}/{NUM_ROUNDS} ---")

            selected_client_ids = np.random.choice(
                range(NUM_CLIENTS), num_selected_clients, replace=False
            )
            print(f"Selected clients: {selected_client_ids}")

            client_deltas = []

            # Client training phase
            for client_id in selected_client_ids:
                dataloader = client_dataloaders[client_id]

                delta = arm2_perturb_client_update(
                    client_id, global_model, dataloader,
                    LOCAL_EPOCHS, r, writer,
                    TARGET_EPSILON, L2_SENSITIVITY
                )

                if delta is not None:
                    client_deltas.append(delta)

            # Server aggregation phase
            if client_deltas:
                server_aggregate(global_model, client_deltas)

            # Global model evaluation phase
            test_model(global_model, test_loader, r, writer)

            print(f"Round {r} completed in {time.time() - start_time:.2f}s")

    print(f"\nOne-Shot training complete. Log saved to {LOG_FILE}")


# --- Run the experiment ---
if __name__ == "__main__":
    run_arm2_experiment()
