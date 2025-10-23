import torch
import evaluate
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
OUTPUT_DIR = "./flan-t5-small-biolaysumm-manual-loop"

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

model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader
)

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
        
        all_predictions.append(accelerator.gather_for_metrics(generated_tokens))
        all_labels.append(accelerator.gather_for_metrics(batch["labels"]))
    
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    all_labels = torch.where(all_labels != -100, all_labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(all_predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(all_labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: round(value * 100, 2) for key, value in result.items()}

    accelerator.print(f"Epoch {epoch+1} Validation ROUGE: {result}")

accelerator.print("--- Training Complete. Evaluating on Test Set ---")
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
    
    all_test_predictions.append(accelerator.gather_for_metrics(generated_tokens))
    all_test_labels.append(accelerator.gather_for_metrics(batch["labels"]))

# Concatenate all gathered tensors
all_test_predictions = torch.cat(all_test_predictions)
all_test_labels = torch.cat(all_test_labels)

# Decode
all_test_labels = torch.where(all_test_labels != -100, all_test_labels, tokenizer.pad_token_id)
decoded_test_preds = tokenizer.batch_decode(all_test_predictions, skip_special_tokens=True)
decoded_test_labels = tokenizer.batch_decode(all_test_labels, skip_special_tokens=True)

# Compute final ROUGE scores
final_result = rouge.compute(predictions=decoded_test_preds, references=decoded_test_labels, use_stemmer=True)
final_result = {key: round(value, 4) for key, value in final_result.items()}

accelerator.print("\n--- FINAL TEST SET RESULTS ---")
accelerator.print(final_result)


# Only the main process should handle saving the model and results
if accelerator.is_main_process:

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    accelerator.print(f"\n--- Peak VRAM Usage: {peak_vram_gb:.2f} GB ---")
    accelerator.print(f"Saving model to {OUTPUT_DIR}...")
    # Ensure everyone is done before saving
    accelerator.wait_for_everyone()
    
    # unwrap_model gets the base model back from the Accelerator
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save final results to a JSON file
    results_path = os.path.join(OUTPUT_DIR, RESULTS_FILE)
    with open(results_path, "w") as f:
        json.dump(final_result, f, indent=2)
    
    accelerator.print(f"Final results saved to {results_path}")

accelerator.print("--- Script Finished ---")