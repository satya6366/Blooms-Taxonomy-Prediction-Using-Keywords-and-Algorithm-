import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import os

# Ensure necessary packages are installed

# Function to get user input for semester
def get_semester():
    while True:
        try:
            semester = int(input("Enter the semester (1-8): "))
            if 1 <= semester <= 8:
                return semester
            else:
                print("Please enter a number between 1 and 8.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Load dataset
data = pd.read_csv('bloom_data.csv')

# Get user input for semester
semester = get_semester()

# Filter data for the selected semester
data = data[data['semester'] == semester]

if data.empty:
    print(f"No data available for semester {semester}.")
else:
    # Preprocessing
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    labels = data['label'].unique().tolist()
    label_map = {label: idx for idx, label in enumerate(labels)}

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)

    data['label'] = data['label'].map(label_map)

    # Split dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(data['text'], data['label'], test_size=0.2)

    # Tokenize
    train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)

    class BloomDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = BloomDataset(train_encodings, train_labels.tolist())
    val_dataset = BloomDataset(val_encodings, val_labels.tolist())

    # Model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels))

    # Training
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    # Function to classify new text
    def classify_text(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
        return labels[predictions.item()]

    # Example usage
    new_text = "write a javascript program to modify html elements?."
    bloom_level = classify_text(new_text)
    print(f"The Bloom's level for the given text is: {bloom_level}")
