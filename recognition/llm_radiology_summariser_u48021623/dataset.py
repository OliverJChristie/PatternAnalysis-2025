# dataset.py
# Contains the data loader for loading and preprocessing the data

import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq

def get_tokenized_datasets(tokenizer, model_checkpoint):
    """
    Loads and preprocesses the BioLay-Summ dataset.
    """
    # Load the dataset from Hugging Face
    dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")

    prefix = "translate Radiology to Layperson: "

    def preprocess_function(examples):
        """ Tokenizes the input and target texts. """
        # 'radiology_report' is the expert report, 'layman_report' is the target
        inputs = [prefix + doc for doc in examples["radiology_report"]]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)

        # Tokenize the target summaries
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["layman_report"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    print("Mapping and tokenizing datasets...")

    # Apply the preprocessing function to all splits
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Create a data collator to handle padding
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_checkpoint)

    return tokenized_datasets, data_collator
