import os
import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Enable CUDA debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Load the Sentiment140 dataset
dataset = load_dataset("sentiment140")

# Load DistilBERT Tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Map sentiment labels to binary: 0 -> 0 (negative), 4 -> 1 (positive)
def convert_labels(examples):
    examples['labels'] = [1 if sentiment == 4 else 0 for sentiment in examples['sentiment']]  # Ensure 4 -> 1 and others -> 0
    return examples

# Apply the transformation
tokenized_datasets = tokenized_datasets.map(convert_labels, batched=True)

# Sanity check: Print unique labels to ensure they are correct
print(set([label for label in tokenized_datasets["train"]["labels"]]))  # Should only be {0, 1}

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Updated from `evaluation_strategy`
    per_device_train_batch_size=4,  # Reduced batch size for debugging
    per_device_eval_batch_size=8,   # Reduced batch size
    num_train_epochs=3,
    learning_rate=1e-5,
    save_strategy="epoch",  # Saves a checkpoint at the end of each epoch
    save_total_limit=2,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2),
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,  # Note: tokenizer argument is deprecated, can be removed in future versions
)

# Start training
trainer.train()
