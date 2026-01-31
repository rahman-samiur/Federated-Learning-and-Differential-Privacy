"""flower-test: A Flower / PyTorch app."""
import os
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
# --- New import ---
from flwr.clientapp.mod import fixedclipping_mod
# ------------------

from flower_test.task import Net, load_data
from flower_test.task import test as test_fn
from flower_test.task import train as train_fn

# Flower ClientApp
# --- REMOVE the mod from here ---
app = ClientApp()
# --------------------------------


# --- ADD the mod to the @app.train decorator ---
@app.train(mods=[fixedclipping_mod])
# @app.train()  # No mods! Privacy is on the server.
def train(msg: Message, context: Context):
    """Train the model on local data."""
    current_round = msg.content["config"]["server-round"]
    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )
    save_dir = "client_weights"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, f"{current_round}_client_{partition_id}_weights.pt")
    
    torch.save(model.state_dict(), save_path)
    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


# --- No mod needed for evaluate ---
@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
