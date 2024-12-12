from transformers import BertTokenizerFast, BertForTokenClassification
import os

MODEL_NAME = "Evheniia/bert_ner"
MODEL_DIR = "./model_save"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print("Downloading the model and tokenizer...")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
model = BertForTokenClassification.from_pretrained(MODEL_NAME)

print(f"Saving model and tokenizer to {MODEL_DIR}...")
tokenizer.save_pretrained(MODEL_DIR)
model.save_pretrained(MODEL_DIR)

print("Model and tokenizer successfully downloaded and saved!")

