import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dataset import get_raw_test_examples

MODEL_DIR = "./flan-t5-base-biolaysumm-manual-loop"
NUM_EXAMPLES = 5

print(f"Loading fine-tuned model from: {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded onto device: {device}")

example_reports = get_raw_test_examples(num_examples=NUM_EXAMPLES, seed=42)

prefix = "translate Radiology to Layperson: "

print("\n--- Generating Layperson Summaries (using num_beams=4) ---")

for i,report in enumerate(example_reports):
    text_to_translate = prefix + report
    inputs = tokenizer(text_to_translate, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

    decoded_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\n--- EXAMPLE {i+1} ---")
    print(f"EXPERT REPORT: \n{report}")
    print(f"\nLAYPERSON SUMMARY:\n{decoded_summary}")

print(f"\n--- Prediction Complete ---")