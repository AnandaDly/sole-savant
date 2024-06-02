import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Memuat data sepatu
shoe_data = pd.read_csv('../data/Shoes_Data_Final.csv')

# Menggabungkan semua review menjadi satu kolom
shoe_data['combined_reviews'] = shoe_data[['review_1', 'review_2', 'review_3', 'review_4', 'review_5',
                                            'review_6', 'review_7', 'review_8', 'review_9', 'review_10']].fillna('').agg(' '.join, axis=1)

# Encode sentiment labels
label_encoder = LabelEncoder()
shoe_data['encoded_sentiment'] = label_encoder.fit_transform(shoe_data['sentiment'])

# Tokenization and Padding
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(shoe_data['combined_reviews'])
sequences = tokenizer.texts_to_sequences(shoe_data['combined_reviews'])
padded_sequences = pad_sequences(sequences, padding='post', maxlen=200)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, shoe_data['encoded_sentiment'], test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=200),
    LSTM(64, return_sequences=True),
    Dropout(0.5),
    LSTM(32),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # Assuming 3 classes for sentiment: negative, neutral, positive
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=250, validation_data=(X_test, y_test), batch_size=64)

# Save the trained model
model.save('sentiment_model1.h5')

print("Model trained and saved as 'sentiment_model.h5'")
