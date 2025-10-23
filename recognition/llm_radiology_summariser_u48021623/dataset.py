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
    dataset = load_dataset("biolaysumm/biolaysumm2024", "subtask2.1_rag")

    prefix = "translate Radiology to Layperson: "

    def preprocess_function(examples):
        """ Tokenizes the input and target texts. """
        # 'summary' is the expert report, 'lay_summary' is the target
        inputs = [prefix + doc for doc in examples["summary"]]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)

        # Tokenize the target summaries
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["lay_summaru"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    print("Mapping and tokenizing datasets...")

    # Apply the preprocessing function to all splits
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Create a data collator to handle padding
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_checkpoint)

    return tokenized_datasets, data_collator
