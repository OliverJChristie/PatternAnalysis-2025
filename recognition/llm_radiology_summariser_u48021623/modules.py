from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_model_and_tokenizer(model_checkpoint):
    """
    Loads the pretrained model and tokenizer.
    This function acts as the 'model component'
    """
    print(f"Loading model and tokenizer from checkpoint: {model_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    return model, tokenizer