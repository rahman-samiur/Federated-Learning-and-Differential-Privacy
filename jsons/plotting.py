import pandas as pd
import json
import matplotlib.pyplot as plt

# Define file names and labels for the plots
files = [
    ('evaluation_log_serverdp-clip05nz1.jsonl', 'Server DP - Clip 0.5, Noise 1'),
    ('evaluation_log_serverdp0clip05nz3.jsonl', 'Server DP - Clip 0.5, Noise 3'),
    ('evaluation_log_serverdpclip05nz2.jsonl', 'Server DP- clip 0.5, Noise 2'),
    ('evaluation_log.jsonl', 'No DP')
]


def read_jsonl_file(filepath, label):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                # Load one JSON object per line
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                # Handle cases where a line might not be a valid JSON object
                print(
                    f"Error decoding JSON on line from {filepath}: {line.strip()}. Error: {e}")
                pass
    df = pd.DataFrame(data)
    df['Source'] = label
    return df


# Load and combine data from all files
all_data = []
for file_name, label in files:
    try:
        df = read_jsonl_file(file_name, label)
        all_data.append(df)
    except FileNotFoundError:
        # File not found
        continue

if all_data:
    df_combined = pd.concat(all_data, ignore_index=True)

    # Plotting
    plt.style.use('ggplot')
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    # --- Accuracy Plot ---
    ax1 = axes[0]
    for source in df_combined['Source'].unique():
        data = df_combined[df_combined['Source'] == source]
        ax1.plot(data['round'], data['eval_acc'], label=source,
                 marker='o', markersize=3, linewidth=2)

    ax1.set_title('Evaluation Accuracy over Rounds', fontsize=14)
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(title='Source', loc='lower right')
    ax1.grid(True)
    ax1.set_ylim(bottom=0)

    # --- Loss Plot ---
    ax2 = axes[1]
    for source in df_combined['Source'].unique():
        data = df_combined[df_combined['Source'] == source]
        ax2.plot(data['round'], data['eval_loss'], label=source,
                 marker='o', markersize=3, linewidth=2)

    ax2.set_title('Evaluation Loss over Rounds', fontsize=14)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(title='Source', loc='upper right')
    ax2.grid(True)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()

    # Save the plot
    plot_filename = 'evaluation_metrics_plot.png'
    plt.savefig(plot_filename)
