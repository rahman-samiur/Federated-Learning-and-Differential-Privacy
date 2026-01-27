"""flower-fl-baseline: A Flower / PyTorch app."""
from flwr.common.typing import Scalar
from typing import Dict
import numpy as np  
import pickle
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from datasets import load_dataset
from flwr.app import ArrayRecord, MetricRecord
from flwr.serverapp.strategy import FedAvg
import json
from flower_fl_baseline.task import Net
from flwr.common.typing import FitRes


def flatten_array_record(arr_rec: ArrayRecord) -> np.ndarray:
    """Flattens a Flower ArrayRecord (state dict) into a single 1D NumPy vector."""
    # .to_numpy() gives a dict[str, np.ndarray]
    state_dict = arr_rec.to_numpy()
    # Flatten each array in the state dict and concatenate them all
    all_params = [arr.flatten() for arr in state_dict.values()]
    return np.concatenate(all_params)


# --- STEP 2: DEFINE CUSTOM STRATEGY ---
class FedAvgLogWeights(FedAvg):
    """
    A custom FedAvg strategy that logs the flattened weight vector
    from each client in each 'fit' round.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A list to store all client weight vectors for later visualization
        self.all_client_weight_vectors: list[dict] = []
        # <-- Check if init runs
        print("--- FedAvgLogWeights strategy initialized ---")

    def aggregate_fit(
        self,
        round_num: int,
        results: list[tuple[FitRes, Context]],
        failures: list[BaseException],
        context: Context,
    ) -> tuple[ArrayRecord | None, ArrayRecord]:

        # --- Check if we got any results ---
        if not results:
            print(
                f"Round {round_num} [aggregate_fit]: Received 0 client results. No weights saved.")
        else:
            print(
                f"Round {round_num} [aggregate_fit]: Received {len(results)} client results. Processing...")

            # --- Intercept client weights *before* aggregation ---
            for fit_res, ctx in results:
                # Get client ID from context
                node_id = ctx.node_config.get("partition-id", "unknown")

                if node_id == "unknown":
                    print(
                        f"Round {round_num} [aggregate_fit]: Warning! Could not find 'partition-id' in client context.")
                    # Use a placeholder if you still want to save
                    save_id = -1
                else:
                    save_id = int(node_id)

                # Flatten the weights (ArrayRecord) into a 1D vector
                flattened_weights = flatten_array_record(fit_res.parameters)

                # Store the data
                self.all_client_weight_vectors.append({
                    "round": round_num,
                    "node_id": save_id,
                    "weights_vector": flattened_weights
                })

            print(
                f"Round {round_num} [aggregate_fit]: Successfully saved {len(results)} weight vectors.")
            print(
                f"Total vectors saved so far: {len(self.all_client_weight_vectors)}")

        # Call the original FedAvg aggregation logic
        return super().aggregate_fit(round_num, results, failures, context)
# Create ServerApp
app = ServerApp()



@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())
    with open("evaluation_log.jsonl", "w") as f:
        pass
    # Initialize FedAvg strategy
    strategy = FedAvgLogWeights(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )
    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )
    with open("evaluation_log.jsonl", "a") as f:

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

    with open("all_client_weights.pkl", "wb") as f:
        pickle.dump(strategy.all_client_weight_vectors, f)

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pth")
