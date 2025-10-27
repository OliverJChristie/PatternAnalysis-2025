import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

HISTORY_JSON_PATH = "./flan-t5-base-biolaysumm-manual-loop/training_history.json"
PLOT_DIR = "./plots"

def generate_plots(history_data, plot_dir):
    """
    Generates and saves plots for loss and ROUGE scores.
    """

    try:
        train_losses = history_data["train_loss"]
        val_rouge_scores = history_data["val_rouge_scores"]
    except KeyError:
        print("ERROR: History data is missing 'train_loss' or 'val_rouge_scores'.")
        return

    os.makedirs(plot_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10,5))
    plt.plot(epochs, train_losses, 'bo-', label='Average Training Loss')
    plt.title('Training Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.grid(True)
    plt.legend()
    loss_filename = os.path.join(plot_dir, 'training_loss.png')
    plt.savefig(loss_filename)
    print(f"Saved training loss plot to: {loss_filename}")
    plt.close()

    val_rouge = {
        'rouge1': [d['rouge1'] for d in val_rouge_scores],
        'rouge2': [d['rouge2'] for d in val_rouge_scores],
        'rougeL': [d['rougeL'] for d in val_rouge_scores],
        'rougeLsum': [d['rougeLsum'] for d in val_rouge_scores]
    }

    plt.figure(figsize=(10,5))
    # Plot scores as 0-100
    plt.plot(epochs, val_rouge['rouge1'], 'ro-', label='ROUGE-1')
    plt.plot(epochs, val_rouge['rouge2'], 'go-', label='ROUGE-2')
    plt.plot(epochs, val_rouge['rougeL'], 'bo-', label='ROUGE-L')
    plt.plot(epochs, val_rouge['rougeLsum'], 'mo-', label='ROUGE-Lsum')

    plt.title('Validation ROUGE Scores vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylim([0, 100])
    plt.ylabel('ROUGE Score (0-100)')
    plt.xticks(epochs)
    plt.yticks(range(0, 101, 20))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(5))
    plt.grid(which='major', linestyle='-', linewidth=0.8)
    plt.grid(which='minor', linestyle='--', linewidth=0.5)

    plt.legend()
    rouge_filename = os.path.join(plot_dir, 'validation_rouge.png')
    plt.savefig(rouge_filename)
    print(f"Saved validation ROUGE plot to: {rouge_filename}")
    plt.close()

if __name__ == "__main__":
    try:
        with open(HISTORY_JSON_PATH, 'r') as f:
            history = json.load(f)
        generate_plots(history, PLOT_DIR)
    except FileNotFoundError:
        print(f"ERROR: Could not find {HISTORY_JSON_PATH}")
        print(f"Please run train.py first to generate the history file.")
    except Exception as e:
        print(f"An error occurred: {e}")