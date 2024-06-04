import pandas as pd

# Load data from CSV
df = pd.read_csv('../data/Shoes_Data_Final.csv')

# Melt DataFrame to long format
reviews = df.melt(value_vars=[f'review_{i+1}' for i in range(10)], var_name='review', value_name='text')
ratings = df.melt(value_vars=[f'review_rating_{i+1}' for i in range(10)], var_name='rating', value_name='rating_text')

# Combine reviews and ratings
combined_df = pd.DataFrame({'review': reviews['text'], 'rating': ratings['rating_text']})

# Convert ratings to numeric values
combined_df['rating'] = combined_df['rating'].str.extract(r'(\d.\d)').astype(float)

# Drop NaN values
combined_df.dropna(inplace=True)

# Labeling sentiment based on rating
def label_sentiment(rating):
    if rating >= 4.0:
        return 'positive'
    elif rating <= 2.0:
        return 'negative'
    else:
        return 'neutral'

combined_df['sentiment'] = combined_df['rating'].apply(label_sentiment)

# Encode labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
combined_df['label'] = label_encoder.fit_transform(combined_df['sentiment'])

# Split data into features and labels
X = combined_df['review']
y = combined_df['label']

from transformers import BertTokenizer

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization function
def tokenize_function(texts):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors="pt")

# Tokenize the dataset
from sklearn.model_selection import train_test_split
train_texts, test_texts, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)
train_encodings = tokenize_function(train_texts.tolist())
test_encodings = tokenize_function(test_texts.tolist())

#!pip install transformers[torch] accelerate -U

from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Dataset Class
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels.tolist())
test_dataset = SentimentDataset(test_encodings, test_labels.tolist())

# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size for training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

# Trainer
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset             # evaluation dataset
)

# Train the model
trainer.train()

import os
# Ensure the model directory exists
if not os.path.exists('model'):
    os.makedirs('model')

# Save the model to the "model" directory
model.save_pretrained("model")
tokenizer.save_pretrained("model")