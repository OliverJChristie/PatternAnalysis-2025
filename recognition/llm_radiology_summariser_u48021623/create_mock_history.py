"""
Parses SLURM log files to extract/save metrics to a JSON file.
Reimplements behaviour from train.py, and not required if train.py is run.
"""

import re
import json
import os
import numpy as np
import ast

LOG_FILE_PATH = "./outputs/llm_full_multi_gpu_321123.out"
OUTPUT_DIR = "./flan-t5-base-biolaysumm-manual-loop"
HISTORY_FILE = "training_history.json"

def parse_log_to_json(log_path, output_dir, history_file):
    """Parses the SLURM log file and saves metrics to a JSON file."""
    train_losses = []
    val_rouge_scores = []
    
    # Regex patterns to find the lines
    loss_pattern = re.compile(r"Epoch \d+ Average Train Loss: ([\d\.]+)")
    rouge_pattern = re.compile(r"Epoch \d+ Validation ROUGE: (\{.*\})")

    print(f"Opening log file: {log_path}")
    try:
        with open(log_path, 'r') as f:
            for line in f:
                loss_match = loss_pattern.search(line)
                if loss_match:
                    train_losses.append(float(loss_match.group(1)))
                    
                rouge_match = rouge_pattern.search(line)
                if rouge_match:
                    dict_str = rouge_match.group(1)
                    
                    # Clean the string for safe evaluation
                    # Removes "np.float64(" and ")"
                    cleaned_str = dict_str.replace("np.float64(", "").replace(")", "")
                    
                    # Evaluate the string as a Python literal (dictionary)
                    try:
                        rouge_dict = ast.literal_eval(cleaned_str)
                        val_rouge_scores.append(rouge_dict)
                    except Exception as e:
                        print(f"Warning: Could not parse ROUGE dictionary: {e}")
                        
    except FileNotFoundError:
        print(f"ERROR: Log file not found at {log_path}")
        return

    if not train_losses or not val_rouge_scores:
        print("ERROR: Could not find any loss or ROUGE data in the log.")
        return

    training_history = {
        "train_loss": train_losses,
        "val_rouge_scores": val_rouge_scores
    }

    os.makedirs(output_dir, exist_ok=True)
    
    # Save the history as a JSON file
    history_path = os.path.join(output_dir, history_file)
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
        
    print(f"Successfully created mock history file at: {history_path}")
    print(json.dumps(training_history, indent=2))

if __name__ == "__main__":
    parse_log_to_json(LOG_FILE_PATH, OUTPUT_DIR, HISTORY_FILE)