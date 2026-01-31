"""flower-test: A Flower / PyTorch app."""
import os
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
# --- NEW: Only import FedAvg (Server DP removed) ---
from flwr.serverapp.strategy import FedAvg
import json

from flower_test.task import Net

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # --- Read DP config for logging/naming only ---
    clipping_norm: float = context.run_config["clipping-norm"]
    # We also read epsilon, delta, and sensitivity for naming,
    # but only clipping_norm is varied in the grid search
    epsilon: float = context.run_config.get("epsilon", 10.0)
    # -----------------------------------------------

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy (No DP wrapping needed)
    strategy = FedAvg(fraction_train=fraction_train)

    # --- UNIQUE LOG & MODEL FILENAME IMPLEMENTATION ---
    # Convert DP values to strings for filename (replace '.' with 'p')
    cn_str = str(clipping_norm).replace('.', 'p')
    eps_str = str(epsilon).replace('.', 'p')

    # Construct the unique log filename for CLIENT DP
    log_filename = f"evaluation_log_clientdp_cn{cn_str}_eps{eps_str}.jsonl"
    # Construct the unique model filename for CLIENT DP
    model_filename = f"final_model_clientdp_cn{cn_str}_eps{eps_str}.pt"
    # --------------------------------------------------

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # --- Logging section ---
    with open(log_filename, "w") as f:
        pass
    with open(log_filename, "a") as f:
        for round_num, metrics in result.evaluate_metrics_clientapp.items():
            log_entry = {
                "round": round_num,
                "run_id": str(context.run_id),
                **metrics,
            }
            f.write(json.dumps(log_entry) + "\n")

    print(f"Metrics logged to {log_filename}")

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, model_filename)
    print(f"Final model saved to {model_filename}")
