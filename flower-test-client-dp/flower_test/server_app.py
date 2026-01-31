"""flower-test: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
# --- New import ---
from flwr.serverapp.strategy import DifferentialPrivacyServerSideFixedClipping, FedAvg
import json
from flwr.serverapp.strategy import (
    DifferentialPrivacyClientSideFixedClipping,
    FedAvg,
)
# ------------------

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
    # --- Read new DP config ---
    noise_multiplier: float = context.run_config["noise-multiplier"]
    clipping_norm: float = context.run_config["clipping-norm"]
    num_sampled_clients: int = context.run_config["num-sampled-clients"]
    # --------------------------
    with open("evaluation_log_client_dp.jsonl, "w") as f:
        pass
    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=fraction_train)

    # --- Wrap the strategy with the DP wrapper ---
    dp_strategy = DifferentialPrivacyClientSideFixedClipping(
        strategy,
        noise_multiplier=noise_multiplier,
        clipping_norm=clipping_norm,
        num_sampled_clients=num_sampled_clients,
    )
    # -------------------------------------------
    # dp_strategy = DifferentialPrivacyServerSideFixedClipping(
    #     strategy,  # <--- Pass FedAvg in
    #     noise_multiplier=noise_multiplier,
    #     clipping_norm=clipping_norm,
    #     num_sampled_clients=num_sampled_clients,
    # )
    # Start strategy, run FedAvg for `num_rounds`
    # --- Use the DP strategy ---
    result = dp_strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )
    # ---------------------------
    with open("evaluation_log_client_dp.jsonl", "a") as f:

        # --- THE FIX ---
        # Change `result.metrics_centralized` to `result.evaluate_metrics_clientapp`
        for round_num, metrics in result.evaluate_metrics_clientapp.items():
            # -----------------

            # This part remains the same. The `metrics` object is a
            # MetricRecord, which can be unpacked like a dict.
            log_entry = {
                "round": round_num,
                "run_id": str(context.run_id),
                **metrics,  # Unpack all metrics (eval_loss, eval_acc, etc.)
            }
            # Write each round's metrics as a new line
            f.write(json.dumps(log_entry) + "\n")

    print("Metrics logged to evaluation_log.jsonl")
    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
