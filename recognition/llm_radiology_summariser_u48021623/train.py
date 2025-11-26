import torch
import evaluate
import os
import json
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm.auto import tqdm

from modules import get_model_and_tokenizer
from dataset import get_tokenized_datasets
from plot_results import generate_plots

# --- Hyperparameters & Configuration
MODEL_CHECKPOINT = "google/flan-t5-base"
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
OUTPUT_DIR = "./flan-t5-base-biolaysumm-manual-loop"
RESULTS_FILE = "final_results_base.json"
HISTORY_FILE = "training_history.json"
PLOT_DIR = "./plots"

# Initialise Accelerator
# This handles device placement (GPU selection) and Mixed Precision (bf16) automatically
accelerator = Accelerator()
accelerator.print(f"Using Accelerator state: {accelerator.state}")

# --- Load Resources ---
accelerator.print("Loading model, tokenizer, and dataset...")
model, tokenizer = get_model_and_tokenizer(MODEL_CHECKPOINT)
tokenized_datasets, data_collator = get_tokenized_datasets(tokenizer, model)
rouge = evaluate.load("rouge")

# --- DataLoaders ---
# We use the data_collator to dynamically pad batches to the maximum length in the batch
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=TRAIN_BATCH_SIZE
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    collate_fn=data_collator,
    batch_size=EVAL_BATCH_SIZE
)
test_dataloader = DataLoader(
    tokenized_datasets["test"],
    collate_fn=data_collator,
    batch_size=EVAL_BATCH_SIZE
)
# --- Optimiser & Scheduler ---
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

num_training_steps = NUM_EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# Prepare everything with accelerator
# This wraps the model, optimiser, and dataloaders to handle device placement
# and mixed precision casting automatically
model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler
)

# This will store our best validation score
best_validation_results = {}
training_history = {"train_loss": [], "val_rouge_scores": []}

accelerator.print("--- Starting Manual Training Loop ---")
for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    model.train()
    train_loss = 0

    train_progress_bar = tqdm(
        train_dataloader,
        disable=not accelerator.is_local_main_process,
        desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"
    )

    for batch in train_progress_bar:
        # Forward pass
        outputs = model(**batch)
        # The model computes loss internally because 'labels' are provided in the batch
        loss = outputs.loss

        train_loss += loss.item()

        # Backward pass using accelerator
        # (This handles loss scaling necessary for mixed precision training)
        accelerator.backward(loss)

        # Gradient Clipping
        # Essential to prevent "exploding gradients" which cause nan loss
        accelerator.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        train_progress_bar.set_postfix(loss=loss.item())
    
    avg_train_loss = train_loss / len(train_dataloader)
    accelerator.print(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.4f}")

    training_history["train_loss"].append(avg_train_loss)

    # --- Validation Phase ---
    accelerator.print(f"--- Starting Evaluation for Epoch {epoch+1} ---")
    model.eval()

    all_predictions = []
    all_labels = []

    eval_progress_bar = tqdm(
        eval_dataloader,
        disable=not accelerator.is_local_main_process,
        desc="Evaluating"
    )

    for batch in eval_progress_bar:
        with torch.no_grad():
            # Unwrap model to access the underlying .generate() method
            # and generate text summaries
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128
            )

        # Gather tokens from all processes (GPUs)
        # This ensures we evaluate on the full validation set even in distributed scenarios
        gathered_tokens = accelerator.gather_for_metrics(generated_tokens)
        gathered_labels = accelerator.gather_for_metrics(batch["labels"])

        # Replace -100 in the labels as we can't decode them (tokenizer throws error)
        gathered_labels = torch.where(gathered_labels != -100, gathered_labels, tokenizer.pad_token_id)

        # Decode tokens to text
        decoded_preds = tokenizer.batch_decode(gathered_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(gathered_labels, skip_special_tokens=True)
        
        all_predictions.extend(decoded_preds)
        all_labels.extend(decoded_labels)

    # Compute ROUGE metrics
    result = rouge.compute(predictions=all_predictions, references=all_labels, use_stemmer=True)
    # Convert numpy floats to standard Python floats for JSON serialization
    result_serializable = {key: float(value) for key, value in result.items()}

    accelerator.print(f"Epoch {epoch+1} Validation ROUGE: {best_validation_results}")

    training_history["val_rouge_scores"].append(result_serializable)
    best_validation_results = result_serializable

# --- Test Phase ---
accelerator.print("--- Training Complete. Evaluating on (blind) Test Set ---")
# NOTE: The test set is blind (no labels), so we cannote compute ROUGE scores.
# This loop's purpose is to generate predictions for the entire test set
# to ensure the model runs without errors on the final unseen data.
# For a few visual examples with inputs and outputs, run predict.py.
model.eval()

all_test_predictions = []
all_test_labels = []

test_progress_bar = tqdm(
    test_dataloader, 
    disable=not accelerator.is_local_main_process,
    desc="Running Final Test"
)

for batch in test_progress_bar:
    with torch.no_grad():
        generated_tokens = accelerator.unwrap_model(model).generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=128
        )
    
    gathered_tokens = accelerator.gather_for_metrics(generated_tokens)

    decoded_preds = tokenizer.batch_decode(gathered_tokens, skip_special_tokens=True)

    all_test_predictions.extend(decoded_preds)

accelerator.print(f"\n--- Test set prediction generation complete. (Test set is blind, ROUGE=0.0) ---")
accelerator.print(f"--- Using Epoch 3 Validation ROUGE for final results. ---")

# --- Saving Artifacts ---
# Only the main process should handle saving the model and results to avoid file corruption
if accelerator.is_main_process:

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    accelerator.print(f"\n--- Peak VRAM Usage: {peak_vram_gb:.2f} GB ---")
    accelerator.print(f"Saving model to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Ensure everyone is done before saving
    accelerator.wait_for_everyone()
    
    # Unwrap model to save the clean underlying model (removes Accelerator wrappers)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save final results to a JSON file
    results_path = os.path.join(OUTPUT_DIR, RESULTS_FILE)
    final_results_rounded = {key: round(value, 4) for key, value in best_validation_results.items()}
    with open(results_path, "w") as f:
        json.dump(best_validation_results, f, indent=2)
    accelerator.print(f"Final results saved to {results_path}")

    # Save training history
    history_path = os.path.join(OUTPUT_DIR, HISTORY_FILE)
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    accelerator.print(f"Training history saved to {history_path}")

    # Generate plots
    accelerator.print(f"Generating plots in {PLOT_DIR}")
    try:
        generate_plots(training_history, PLOT_DIR)
        accelerator.print("Plots generated successfully.")
    except Exception as e:
        accelerator.print(f"ERROR: Cound not generate plots: {e}")

accelerator.print("--- Script Finished Successfully ---")
