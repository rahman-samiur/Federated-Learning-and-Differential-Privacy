import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import csv
import time
from collections import OrderedDict

# --- Opacus import for the ACCOUNTANT ---
from opacus.accountants import RDPAccountant

# --- Imports from baseline_fl.py ---
try:
    from baseline_fl import (
        SimpleCNN,
        get_dataloaders,
        test_model
    )
except ImportError:
    print("Error: Could not import from baseline_fl.py.")
    print("Please ensure baseline_fl.py is in the same directory.")
    exit()

# --- 1. Client Update (per Geyer et al. Algorithm 1) ---
# (This function is unchanged)


def client_update(client_id, model, dataloader, epochs, round_num, csv_writer):
    """
    Implements the ClientUpdate function from the paper.
    - Trains locally
    - Returns the *un-clipped* delta AND the L2 norm of the delta
    """
    local_model = copy.deepcopy(model)
    local_model.train()

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

        avg_epoch_loss = epoch_loss / total if total > 0 else 0
        avg_epoch_acc = 100.0 * correct / total if total > 0 else 0

        log_row = [round_num, client_id,
                   f'train_epoch_{e+1}', avg_epoch_loss, avg_epoch_acc]
        csv_writer.writerow(log_row)

    local_weights = local_model.state_dict()
    model_delta = OrderedDict()
    for key in global_weights:
        model_delta[key] = local_weights[key] - global_weights[key]

    flat_delta = torch.cat([p.view(-1) for p in model_delta.values()])
    delta_norm = torch.norm(flat_delta, p=2).item()

    # Handle nan from exploded model
    if np.isnan(delta_norm):
        delta_norm = 0.0  # Send 0 norm if model exploded

    return model_delta, delta_norm

# --- 2. Server Aggregation (CORRECTED per Geyer et al. Algorithm 1) ---


def central_dp_aggregate(global_model, client_results, sigma_hyperparam):
    """
    Implements the ServerExecution aggregation from the paper.
    - Dynamically clips using the median norm.
    - Adds noise (using fixed sigma) to the sum, then averages.
    """
    if not client_results:
        return

    num_clients_K = len(client_results)

    # --- 1. Get Median Norm (S) ---
    all_norms = [res['norm'] for res in client_results]
    l2_sensitivity_S = float(np.median(all_norms))

    # Handle the case where median is 0 or nan
    if l2_sensitivity_S <= 0:
        l2_sensitivity_S = 1.0  # Set a default to prevent division by zero

    print(f"  Central DP: Dynamic S (median norm) = {l2_sensitivity_S:.4f}")

    # --- 2. Clip, Sum, and Add Noise to Sum ---
    sum_delta = OrderedDict()
    for key in client_results[0]['delta']:
        sum_delta[key] = torch.zeros_like(client_results[0]['delta'][key])

    for res in client_results:
        delta = res['delta']
        norm = res['norm']
        clipping_factor = min(1.0, l2_sensitivity_S / (norm + 1e-6))

        for key in sum_delta:
            # Handle nan deltas from clients with exploded models
            if not torch.isnan(delta[key]).any():
                sum_delta[key] += delta[key] * clipping_factor

    # Add noise to the sum (per Algorithm 1)
    noise_std_for_sum = l2_sensitivity_S * sigma_hyperparam

    for key in sum_delta:
        noise = torch.normal(0, noise_std_for_sum, size=sum_delta[key].shape)
        sum_delta[key] += noise

    # --- 3. Average the Noisy Sum ---
    avg_noisy_delta = OrderedDict()
    for key in sum_delta:
        avg_noisy_delta[key] = sum_delta[key] / num_clients_K

    # --- 4. Update Global Model ---
    updated_weights = global_model.state_dict()
    for key in updated_weights:
        if not torch.isnan(avg_noisy_delta[key]).any():
            updated_weights[key] += avg_noisy_delta[key]

    global_model.load_state_dict(updated_weights)

# --- 3. Main Training Loop (CORRECTED) ---


def run_arm3_paper_experiment():

    # --- Hyperparameters ---
    NUM_CLIENTS = 20
    NUM_ROUNDS = 50
    LOCAL_EPOCHS = 3
    BATCH_SIZE = 64
    CLIENT_FRACTION = 0.5

    # --- DP-Specific Parameters ---
    # This is now a STOPPING condition
    TARGET_EPSILON = 2.0
    TARGET_DELTA = 1.0 / 60000

    # This is the hyperparameter 'sigma' from the paper
    SIGMA_HYPERPARAMETER = 1.0

    LOG_FILE = 'arm3_geyer_paper_log.csv'

    print("Setting up Arm 3 (Geyer et al. 2017) experiment...")
    global_model = SimpleCNN()
    client_dataloaders, test_loader = get_dataloaders(
        num_clients=NUM_CLIENTS, iid=True, batch_size=BATCH_SIZE
    )

    num_selected_clients = int(NUM_CLIENTS * CLIENT_FRACTION)

    # --- Set up the Privacy Accountant ---
    accountant = RDPAccountant()

    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'client_id', 'type',
                        'loss', 'accuracy', 'epsilon_spent'])

        print(
            f"Starting Central DP (Target Eps={TARGET_EPSILON}, Sigma={SIGMA_HYPERPARAMETER}, Dynamic S)...")

        test_model(global_model, test_loader, 0, writer)

        for r in range(1, NUM_ROUNDS + 1):
            start_time = time.time()
            print(f"\n--- Round {r}/{NUM_ROUNDS} ---")

            selected_client_ids = np.random.choice(
                range(NUM_CLIENTS), num_selected_clients, replace=False
            )

            client_results = []

            # Client training phase
            for client_id in selected_client_ids:
                dataloader = client_dataloaders[client_id]

                delta, norm = client_update(
                    client_id, global_model, dataloader,
                    LOCAL_EPOCHS, r, writer
                )
                client_results.append({'delta': delta, 'norm': norm})

            # Server aggregation phase
            central_dp_aggregate(
                global_model, client_results,
                SIGMA_HYPERPARAMETER
            )

            # --- Track Privacy ---
            accountant.step(
                noise_multiplier=SIGMA_HYPERPARAMETER,
                sample_rate=CLIENT_FRACTION
            )

            # Get current privacy spent
            epsilon_spent = accountant.get_epsilon(delta=TARGET_DELTA)
            print(
                f"  Privacy Spent (Epsilon): {epsilon_spent:.4f} / {TARGET_EPSILON}")

            # Global model evaluation phase
            test_model(global_model, test_loader, r, writer)

            print(f"Round {r} completed in {time.time() - start_time:.2f}s")

            # --- Check Stopping Condition ---
            if epsilon_spent > TARGET_EPSILON:
                print(
                    f"\nPrivacy budget exhausted. Stopping training at round {r}.")
                break

    print(
        f"\nCentral DP (Geyer et al.) training complete. Log saved to {LOG_FILE}")


if __name__ == "__main__":
    run_arm3_paper_experiment()
