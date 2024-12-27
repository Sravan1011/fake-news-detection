import pandas as pd
import numpy as np
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

train = pd.read_csv('/content/drive/MyDrive/fake-news-detection/train.tsv', sep='\t')
test = pd.read_csv('/content/drive/MyDrive/fake-news-detection/test.tsv', sep='\t')

# Drop index, title, subject, and date columns
train = train.drop(train.columns[0], axis=1)
test = test.drop(test.columns[0], axis=1)

cols_to_drop = ['title', 'subject', 'date']
for col in cols_to_drop:
    if col in train.columns:
        train = train.drop(col, axis=1)
    if col in test.columns:
        test = test.drop(col, axis=1)

# Data Cleaning
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'http\S+', '', text)  # Remove URLs and links
    return text

train['text'] = train['text'].apply(clean_text)
test['text'] = test['text'].apply(clean_text)

# Prepare the data
X_train = train['text']
y_train = train['label']
X_test = test['text']
y_test = test['label']

# Tokenization and padding
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=200)
X_test_pad = pad_sequences(X_test_seq, maxlen=200)

# Model building
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=200),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the model
y_pred = model.predict(X_test_pad)
y_pred_binary = (y_pred > 0.5).astype(int)

# Metrics
accuracy = accuracy_score(y_test, y_pred_binary)
classification_rep = classification_report(y_test, y_pred_binary)

# Save results to a .txt file
with open("results.txt", "w") as file:
    file.write("Model Performance Metrics:\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write("\nClassification Report:\n")
    file.write(classification_rep)
    file.write("\n\nPredictions:\n")
    for i, pred in enumerate(y_pred_binary):
        label = "Original" if pred[0] == 1 else "Fake"
        actual_label = "Original" if y_test.iloc[i] == 1 else "Fake"
        file.write(f"Sample {i+1}:\n")
        file.write(f"Text: {X_test.iloc[i]}\n")
        file.write(f"Predicted: {label}, Actual: {actual_label}\n\n")

print("Results saved to 'results.txt'.")