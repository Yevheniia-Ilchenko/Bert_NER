import torch
from transformers import BertTokenizerFast, BertForTokenClassification

MODEL_DIR = "./model_save"
id2tag = {0: 'O', 1: 'B-MOUNTAIN', 2: 'I-MOUNTAIN'}

tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
model = BertForTokenClassification.from_pretrained(MODEL_DIR)

def predict(text: str):
    words = text.strip().split()


    inputs = tokenizer(words, return_tensors="pt", is_split_into_words=True, padding=True, truncation=True, max_length=512)
    word_ids = inputs.word_ids(batch_index=0)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()

    word_preds = []
    current_word_id = None

    for pred, w_id in zip(predictions, word_ids):
        if w_id is None:
            continue
        if w_id != current_word_id:
            word_preds.append(pred)
            current_word_id = w_id
        else:
            pass

    predicted_labels = [id2tag[p] for p in word_preds]

    return list(zip(words, predicted_labels))



text = "The Dividing Range decreases north of the Kilimanjaro, until as a mere ridge it divides the waters of the coastal rivers from those flowing to the Darling"


token_label_pairs = predict(text)
for token, label in token_label_pairs:
    print(f'{token}: {label}')
print(predict(text))
