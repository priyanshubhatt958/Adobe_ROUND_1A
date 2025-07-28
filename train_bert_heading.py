import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.utils.class_weight import compute_class_weight
import os

# Load data
print('Loading data...')
df = pd.read_csv('bert_training_data.csv')

# Dataset
class HeadingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare dataset
texts = df['bert_input'].tolist()
labels = df['label_id'].tolist()
dataset = HeadingDataset(texts, labels, tokenizer)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print('Class weights:', class_weights.tolist())

# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
model.class_weights = class_weights

def compute_loss(outputs, labels, **kwargs):
    logits = outputs.logits
    loss_fct = torch.nn.CrossEntropyLoss(weight=model.class_weights.to(logits.device))
    loss = loss_fct(logits, labels)
    return loss

# Training arguments
training_args = TrainingArguments(
    output_dir='./saved/bert_heading_model',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=50,
    learning_rate=2e-5,
    weight_decay=0.01,
    report_to=[],
    no_cuda=True,  # CPU only
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    compute_loss_func=compute_loss,
)

# Train
print('Starting training...')
trainer.train()

# Save model
os.makedirs('./saved/bert_heading_model', exist_ok=True)
model.save_pretrained('./saved/bert_heading_model')
tokenizer.save_pretrained('./saved/bert_heading_model')
print('Model and tokenizer saved to ./saved/bert_heading_model') 