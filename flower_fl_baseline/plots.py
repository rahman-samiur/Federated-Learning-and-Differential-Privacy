import json
import os
import re
import glob
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap

# --- Configuration ---
# Set the directory where your .pt files are stored
SAVE_DIR = "client_weights"
# --- End Configuration ---


def flatten_weights(weights):
    """Flattens a state_dict or a tensor into a single 1D numpy array."""
    if isinstance(weights, dict):
        # It's a state_dict, concatenate all tensor parameters
        flat_tensors = []
        for param in weights.values():
            if torch.is_tensor(param):
                flat_tensors.append(param.flatten())
        if not flat_tensors:
            return None
        flat_vector = torch.cat(flat_tensors)
    elif torch.is_tensor(weights):
        # It's a single tensor
        flat_vector = weights.flatten()
    else:
        # Unknown format
        return None

    return flat_vector.detach().cpu().numpy()


def load_and_process_weights(save_dir):
    """Loads all weights from the directory and processes them."""
    print(f"Scanning for weight files in: {save_dir}")

    # Regex to capture round and client ID
    # Format: {current_round}_client_{partition_id}_weights.pt
    pattern = re.compile(r"(\d+)_client_(\d+)_weights\.pt")

    all_vectors = []
    metadata = []  # Store (round, client_id, filename)

    # Use glob to find all matching files
    search_path = os.path.join(save_dir, "*_client_*_weights.pt")
    weight_files = glob.glob(search_path)

    if not weight_files:
        print(f"Error: No weight files found at '{search_path}'.")
        print("Please check your SAVE_DIR variable and file naming convention.")
        return None, None

    print(f"Found {len(weight_files)} weight files. Processing...")

    for filepath in weight_files:
        filename = os.path.basename(filepath)
        match = pattern.match(filename)

        if not match:
            print(f"Skipping {filename}: does not match expected format.")
            continue

        current_round = int(match.group(1))
        partition_id = int(match.group(2))

        try:
            # Load weights, mapping to CPU to avoid GPU memory issues
            weights = torch.load(filepath, map_location=torch.device('cpu'))

            # Flatten the weights into a single vector
            flat_vector = flatten_weights(weights)

            if flat_vector is not None:
                all_vectors.append(flat_vector)
                metadata.append({
                    "round": current_round,
                    "client": partition_id,
                    "filename": filename
                })
            else:
                print(f"Skipping {filename}: could not flatten weights.")

        except Exception as e:
            print(f"Error loading or processing {filename}: {e}")

    if not all_vectors:
        print("Error: No valid weight vectors were processed.")
        return None, None

    print(f"Successfully processed {len(all_vectors)} weight vectors.")

    # Convert list of vectors into a 2D numpy array
    # Each row is a client's flattened weights
    data_matrix = np.stack(all_vectors)

    # Create a pandas DataFrame for metadata (for plotting)
    meta_df = pd.DataFrame(metadata)

    return data_matrix, meta_df


def run_dimensionality_reduction(data_matrix):
    """Runs t-SNE and UMAP on the data matrix."""

    print("Running t-SNE... (This may take a while)")
    # Note: You may need to tune perplexity. A good starting point
    # is often between 5 and 50.
    tsne = TSNE(n_components=2,
                perplexity=30,
                max_iter=1000,
                random_state=42)
    tsne_results = tsne.fit_transform(data_matrix)
    print("t-SNE complete.")

    print("Running UMAP...")
    # Note: You may need to tune n_neighbors and min_dist.
    # n_neighbors controls the balance between local and global structure.
    # min_dist controls how tightly points are clustered.
    umap_model = umap.UMAP(n_components=2,
                           n_neighbors=15,
                           min_dist=0.1,
                           random_state=42)
    umap_results = umap_model.fit_transform(data_matrix)
    print("UMAP complete.")

    return tsne_results, umap_results


def plot_results(tsne_results, umap_results, meta_df):
    """Plots the t-SNE and UMAP results using Seaborn."""

    # Add results to the metadata DataFrame
    plot_df = meta_df.copy()
    plot_df['tsne-1'] = tsne_results[:, 0]
    plot_df['tsne-2'] = tsne_results[:, 1]
    plot_df['umap-1'] = umap_results[:, 0]
    plot_df['umap-2'] = umap_results[:, 1]

    # Set plot style
    sns.set(style='whitegrid', context='notebook',
            rc={'figure.figsize': (18, 8)})

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Create a normalizer for the color map based on round range
    norm = plt.Normalize(plot_df['round'].min(), plot_df['round'].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    # --- t-SNE Plot ---
    #
    sns.scatterplot(
        ax=ax1,
        data=plot_df,
        x='tsne-1',
        y='tsne-2',
        hue='round',
        hue_norm=norm,  # Ensure colors match the colorbar
        palette='viridis',  # Use a sequential colormap for rounds
        s=50,
        alpha=0.7,
        legend=False  # Disable the discrete legend
    )
    ax1.set_title('t-SNE of Client Weights (Hue by Round)', fontsize=16)
    # Add Colorbar
    cbar1 = fig.colorbar(sm, ax=ax1)
    cbar1.set_label('Round')

    # --- UMAP Plot ---
    sns.scatterplot(
        ax=ax2,
        data=plot_df,
        x='umap-1',
        y='umap-2',
        hue='round',
        hue_norm=norm,
        palette='viridis',
        s=50,
        alpha=0.7,
        legend=False
    )
    ax2.set_title('UMAP of Client Weights (Hue by Round)', fontsize=16)
    # Add Colorbar
    cbar2 = fig.colorbar(sm, ax=ax2)
    cbar2.set_label('Round')

    plt.suptitle('Client Weight Distribution (t-SNE & UMAP)',
                 fontsize=20, y=1.02)
    plt.tight_layout()

    # Save the figure
    output_filename = "client_weights_visualization.png"
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

    # Show the plot
    plt.show()
def plot_curves():

    # Initialize an empty list to store the data
    data = []

    # Open the file and read line by line
    # Assuming the file is in the same directory as your script
    with open('evaluation_log.jsonl', 'r') as f:
        for line in f:
            # Parse each line as a separate JSON object
            data.append(json.loads(line))

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)

    # Create a figure with two subplots side-by-side
    plt.figure(figsize=(12, 6))

    # 1. Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(df['round'], df['eval_acc'], label='Accuracy', color='blue')
    plt.title('Evaluation Accuracy per Round')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # 2. Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(df['round'], df['eval_loss'], label='Loss', color='red')
    plt.title('Evaluation Loss per Round')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True)

    # Adjust layout to prevent overlap and save the plot
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    plt.show()

def main():
    data_matrix, meta_df = load_and_process_weights(SAVE_DIR)



    tsne_results, umap_results = run_dimensionality_reduction(data_matrix)

    plot_results(tsne_results, umap_results, meta_df)
    plot_curves()

if __name__ == "__main__":
    main()
