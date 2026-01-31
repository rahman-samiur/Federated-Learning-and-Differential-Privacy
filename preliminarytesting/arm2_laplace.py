import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import csv
import time
from collections import OrderedDict

# --- Imports from baseline_fl.py ---
try:
    from baseline_fl import (
        SimpleCNN,
        get_dataloaders,
        server_aggregate,  # We use the simple baseline aggregator
        test_model
    )
except ImportError:
    print("Error: Could not import from baseline_fl.py.")
    print("Please ensure baseline_fl.py is in the same directory.")
    exit()

# --- Client Update Function (Arm 2: L1/Laplace Perturbation) ---


def arm2_laplace_client_update(client_id, model, dataloader, epochs, round_num, csv_writer,
                               l1_sensitivity_S, per_round_epsilon):
    """
    Implements the Arm 2: One-Shot Model Perturbation using L1/Laplace.
    - Trains locally (non-privately)
    - Clips the *total model delta* L1 norm
    - Adds one-shot calibrated Laplace noise to the delta
    """
    local_model = copy.deepcopy(model)
    local_model.train()

    global_weights = copy.deepcopy(local_model.state_dict())

    optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # --- 1. Standard Non-Private Local Training ---
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

    # --- 3. Clip the Entire Delta (L1 Norm) ---

    # First, compute the total L1 norm of the *entire* delta
    total_l1_norm = 0.0
    for key in model_delta:
        total_l1_norm += torch.sum(torch.abs(model_delta[key]))
    total_l1_norm = total_l1_norm.item()

    print(
        f"  Client {client_id}: Delta L1 norm BEFORE clipping: {total_l1_norm:.2f}")

    # Calculate clipping factor per dp_moon.py
    clipping_factor = min(1.0, l1_sensitivity_S / (total_l1_norm + 1e-6))

    if clipping_factor < 1.0:
        print(
            f"  Client {client_id}: CLIPPING delta from L1 norm {total_l1_norm:.2f} to {l1_sensitivity_S}")
        for key in model_delta:
            model_delta[key] *= clipping_factor

    # --- 4. Add Calibrated Laplace Noise ---

    # The scale 'b' for Laplace noise is Sensitivity / Epsilon
    dp_scale = l1_sensitivity_S / per_round_epsilon
    print(
        f"  Client {client_id}: Adding Laplace noise with scale b = {dp_scale:.4f}")

    # Add Laplace noise to the (now clipped) delta
    for key in model_delta:
        noise = torch.tensor(
            np.random.laplace(0, scale=dp_scale, size=model_delta[key].shape),
            dtype=model_delta[key].dtype
        )
        model_delta[key] += noise

    # Return the noisy, clipped delta
    return model_delta

# --- Main Training Loop ---


def run_arm2_laplace_experiment():
    """
    Orchestrates the Arm 2: L1/Laplace Perturbation experiment.
    """

    # --- Hyperparameters ---
    NUM_CLIENTS = 20
    NUM_ROUNDS = 5
    LOCAL_EPOCHS = 2
    BATCH_SIZE = 64
    CLIENT_FRACTION = 0.5

    L1_SENSITIVITY = 150   # L1 clipping norm (S)
    PER_ROUND_EPSILON = 300.0  # Per-round privacy budget

    LOG_FILE = 'arm2_laplace_log.csv'

    # Setup
    print("Setting up Arm 2: One-Shot (L1/Laplace) experiment...")
    global_model = SimpleCNN()
    client_dataloaders, test_loader = get_dataloaders(
        num_clients=NUM_CLIENTS, iid=True, batch_size=BATCH_SIZE
    )

    num_selected_clients = int(NUM_CLIENTS * CLIENT_FRACTION)

    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'client_id', 'type', 'loss', 'accuracy'])

        print(
            f"Starting L1/Laplace (S={L1_SENSITIVITY}, Eps_per_round={PER_ROUND_EPSILON})...")

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

            # Client training phase (privacy is applied here)
            for client_id in selected_client_ids:
                dataloader = client_dataloaders[client_id]

                delta = arm2_laplace_client_update(
                    client_id, global_model, dataloader,
                    LOCAL_EPOCHS, r, writer,
                    L1_SENSITIVITY, PER_ROUND_EPSILON
                )

                if delta is not None:
                    client_deltas.append(delta)

            # Server aggregation phase (simple averaging)
            if client_deltas:
                server_aggregate(global_model, client_deltas)

            # Global model evaluation phase
            test_model(global_model, test_loader, r, writer)

            print(f"Round {r} completed in {time.time() - start_time:.2f}s")

    print(
        f"\nOne-Shot (L1/Laplace) training complete. Log saved to {LOG_FILE}")


# --- Run the experiment ---
if __name__ == "__main__":
    run_arm2_laplace_experiment()
