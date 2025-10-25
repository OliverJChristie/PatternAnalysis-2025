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

MODEL_CHECKPOINT = "google/flan-t5-base"
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
OUTPUT_DIR = "./flan-t5-base-biolaysumm-manual-loop"
RESULTS_FILE = "final_results_base.json"

accelerator = Accelerator()
accelerator.print(f"Using Accelerator state: {accelerator.state}")

accelerator.print("Loading model, tokenizer, and dataset...")
model, tokenizer = get_model_and_tokenizer(MODEL_CHECKPOINT)
tokenized_datasets, data_collator = get_tokenized_datasets(tokenizer, model)
rouge = evaluate.load("rouge")

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

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

num_training_steps = NUM_EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler
)

# This will store our best validation score
best_validation_results = {}

accelerator.print("--- Starting Manual Training Loop ---")
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0

    train_progress_bar = tqdm(
        train_dataloader,
        disable=not accelerator.is_local_main_process,
        desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"
    )

    for batch in train_progress_bar:
        outputs = model(**batch)
        loss = outputs.loss

        train_loss += loss.item()

        accelerator.backward(loss)

        accelerator.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        train_progress_bar.set_postfix(loss=loss.item())
    
    avg_train_loss = train_loss / len(train_dataloader)
    accelerator.print(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.4f}")

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
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128
            )
        
        gathered_tokens = accelerator.gather_for_metrics(generated_tokens)
        gathered_labels = accelerator.gather_for_metrics(batch["labels"])

        # Decode into strings immediately
        gathered_labels = torch.where(gathered_labels != -100, gathered_labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(gathered_tokens, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(gathered_labels, skip_special_tokens=True)
        
        all_predictions.extend(decoded_preds)
        all_labels.extend(decoded_labels)

    result = rouge.compute(predictions=all_predictions, references=all_labels, use_stemmer=True)
    best_validation_results = {key: round(value * 100, 4) for key, value in result.items()}

    accelerator.print(f"Epoch {epoch+1} Validation ROUGE: {best_validation_results}")

accelerator.print("--- Training Complete. Evaluating on (blind) Test Set ---")
model.eval()

# We still run the test set for predictions, even though we can't score them
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

# Only the main process should handle saving the model and results
if accelerator.is_main_process:

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    accelerator.print(f"\n--- Peak VRAM Usage: {peak_vram_gb:.2f} GB ---")
    accelerator.print(f"Saving model to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Ensure everyone is done before saving
    accelerator.wait_for_everyone()
    
    # unwrap_model gets the base model back from the Accelerator
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save final results to a JSON file
    results_path = os.path.join(OUTPUT_DIR, RESULTS_FILE)
    with open(results_path, "w") as f:
        json.dump(best_validation_results, f, indent=2)
    
    accelerator.print(f"Final results saved to {results_path}")

    example_path = os.path.join(OUTPUT_DIR, "test_set_predictions_examples.txt")
    with open(example_path, "w") as f:
        f.write("--- Example Test Set Predictions (for workshop submission) ---\n\n")
        for i in range(min(5, len(all_test_predictions_list))):
            f.write(f"PREDICTION {i+1}:\n{all_test_predictions_list[i]}\n\n")
            
    accelerator.print(f"Test set prediction examples saved to {example_path}")


accelerator.print("--- Script Finished Successfully ---")