# src/model.py
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_NAME = "bert-base-uncased"

def get_tokenizer():
    return BertTokenizer.from_pretrained(MODEL_NAME)

def get_model():
    return BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )