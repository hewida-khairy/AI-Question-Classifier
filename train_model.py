import json
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import Dataset

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

# ---------------- Simple Data Augmentation ----------------
def simple_augmentation(sentence):
    words = sentence.split()
    if len(words) > 4:
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)

# ---------------- Load Data ----------------
data_path = "E:/aiproject/data.json"
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["question"] for item in data]
labels = [item["label"] for item in data]

# ---------------- Apply Augmentation ----------------
aug_texts = []
aug_labels = []
for text, label in zip(texts, labels):
    aug_texts.append(text)
    aug_labels.append(label)
    for _ in range(2):
        aug_texts.append(simple_augmentation(text))
        aug_labels.append(label)

# ---------------- Encode labels ----------------
unique_labels = sorted(set(aug_labels))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
numeric_labels = [label2id[l] for l in aug_labels]

# ---------------- Train-test split ----------------
train_texts, test_texts, train_labels, test_labels = train_test_split(
    aug_texts, numeric_labels, test_size=0.15, random_state=42, stratify=numeric_labels
)

train_data = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_data = Dataset.from_dict({"text": test_texts, "label": test_labels})

# ---------------- Tokenizer ----------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=128)

train_data = train_data.map(tokenize, batched=True)
test_data = test_data.map(tokenize, batched=True)

# ---------------- Data Collator ----------------
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------- Pretrained Model ----------------
model_pretrained = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)

# ---------------- Model from Scratch ----------------
config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)
model_scratch = AutoModelForSequenceClassification.from_config(config)

# ---------------- Training Arguments ----------------
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_total_limit=1,
    save_steps=50,
    fp16=True
)

# ---------------- Metrics ----------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

# ---------------- Trainer for Pretrained Model ----------------
trainer_pretrained = Trainer(
    model=model_pretrained,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    optimizers=(AdamW(model_pretrained.parameters(), lr=3e-5), None)
)

# ---------------- Train Pretrained Model ----------------
print("Training pretrained model...")
trainer_pretrained.train()
model_pretrained.save_pretrained("model")
tokenizer.save_pretrained("model")

# ---------------- Evaluate Pretrained Model ----------------
result_pretrained = trainer_pretrained.evaluate(eval_dataset=test_data)
print("Pretrained Model Accuracy:", result_pretrained["eval_accuracy"])

# ---------------- Trainer for Scratch Model ----------------
trainer_scratch = Trainer(
    model=model_scratch,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    optimizers=(AdamW(model_scratch.parameters(), lr=3e-5), None)
)

# ---------------- Train Scratch Model ----------------
print("Training model from scratch...")
trainer_scratch.train()
model_scratch.save_pretrained("model_scratch")
tokenizer.save_pretrained("model_scratch")

# ---------------- Evaluate Scratch Model ----------------
result_scratch = trainer_scratch.evaluate(eval_dataset=test_data)
print("Scratch Model Accuracy:", result_scratch["eval_accuracy"])
