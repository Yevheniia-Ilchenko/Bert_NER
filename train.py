
import os
import ast
import torch
import pandas as pd
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support
from typing import Dict, List


MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./model_save"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
EPOCHS = 3


tag2id = {'O': 0, 'B-MOUNTAIN': 1, 'I-MOUNTAIN': 2}
id2tag = {v: k for k, v in tag2id.items()}

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    df['labels'] = df['labels'].apply(ast.literal_eval)
    return df

def align_labels_with_tokens(labels: List[str], word_ids: List[int]) -> List[int]:

    new_labels = [-100] * len(word_ids)
    label_index = 0
    previous_word_id = None

    for i, w_id in enumerate(word_ids):
        if w_id is not None:
            if w_id != previous_word_id:
                new_labels[i] = labels[label_index]
                label_index += 1
            else:
                new_labels[i] = labels[label_index-1]
            previous_word_id = w_id
        else:
            previous_word_id = None
    return new_labels


def prepare_dataset(df: pd.DataFrame, tokenizer: BertTokenizerFast) -> Dict[str, List]:
    tokenized_inputs = tokenizer(
        df['tokens'].tolist(),
        is_split_into_words=True,
        padding='max_length',
        truncation=True,
        return_offsets_mapping=True,
        max_length=MAX_LENGTH
    )
    labels_aligned = []
    for i in range(len(df)):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        labels = [tag2id[tag] for tag in df['labels'][i]]
        aligned = align_labels_with_tokens(labels, word_ids)
        labels_aligned.append(aligned)
    tokenized_inputs['labels'] = labels_aligned
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: Dict[str, List]):
        self.encodings = encodings

    def __getitem__(self, idx: int):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self) -> int:
        return len(self.encodings['input_ids'])


def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(-1)
    true_labels = []
    true_preds = []
    for pred, lab in zip(predictions, labels):
        for p_, l_ in zip(pred, lab):
            if l_ != -100:
                true_labels.append(l_)
                true_preds.append(p_)

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_preds, average='weighted')
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():

    train_df = load_data("train.csv")
    val_df = load_data("val.csv")


    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME, do_lower_case=True)
    model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(tag2id))

    train_encodings = prepare_dataset(train_df, tokenizer)
    val_encodings = prepare_dataset(val_df, tokenizer)

    train_dataset = NERDataset(train_encodings)
    val_dataset = NERDataset(val_encodings)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        learning_rate=LEARNING_RATE,
        save_total_limit=3,
        evaluation_strategy='steps',
        save_strategy='steps',
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )


    trainer.train()


    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


    results = trainer.evaluate()
    print(results)

if __name__ == "__main__":
    main()
