import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os

# Configuration
TEST_MODE = True  # Toggle for testing with limited data
DATA_PATH = 'data.csv'  # Path to CSV file
MODEL_PATH = 'lstm_model.pth'  # Path to save the model
TARGET_COLUMN = 'malicious'  # Target variable for classification
TEST_SIZE = 0.2  # Fraction of data to use for testing
BATCH_SIZE = 32  # Batch size for training
EPOCHS = 10 if not TEST_MODE else 2  # Reduced epochs in test mode
LEARNING_RATE = 0.001  # Learning rate
HIDDEN_SIZE = 64  # LSTM hidden layer size
DROPOUT_RATE = 0.2  # Dropout rate for regularization
RANDOM_STATE = 42  # Random seed
LOAD_EXISTING_MODEL = False  # Toggle to load an existing model if available
SEQUENCE_LENGTH = 10  # Number of time steps per input sequence
WEIGHT_DECAY = 1e-5  # L2 regularization weight decay due to limited scope of time series

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Load and preprocess data
logger.info("Loading data...")
data = pd.read_csv(DATA_PATH)

# Enable test mode with limited data if TEST_MODE is True
if TEST_MODE:
    data = data.sample(n=100, random_state=RANDOM_STATE)  # Limit data to 100 samples
    logger.info("TEST_MODE is ON. Using a limited data subset.")

# Sort by time to preserve chronological order
data = data.sort_values(by='time')

# Define features and target
feature_columns = [col for col in data.columns if col != TARGET_COLUMN]
X = data[feature_columns].values
y = data[TARGET_COLUMN].values

# Convert to sequences
def create_sequences(X, y, sequence_length):
    sequences = []
    labels = []
    for i in range(len(X) - sequence_length + 1):
        sequences.append(X[i:i + sequence_length])
        labels.append(y[i + sequence_length - 1])  # Label at the end of the sequence
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# Prepare sequential data
X_seq, y_seq = create_sequences(X, y, SEQUENCE_LENGTH)

# Split data for training and testing
split_index = int((1 - TEST_SIZE) * len(X_seq))
X_train, X_test = X_seq[:split_index], X_seq[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # hn is the last hidden state
        hn = self.dropout(hn[-1])
        out = self.fc(hn)
        return out

# Model initialization
input_size = X_train.shape[2]
model = LSTMModel(input_size, HIDDEN_SIZE, output_size=2, dropout_rate=DROPOUT_RATE).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Load existing model if toggle is set and file exists
if LOAD_EXISTING_MODEL and os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    logger.info("Loaded existing model from disk.")
else:
    # Training loop
    logger.info("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logger.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss / len(train_loader):.4f}")

    # Save the model after training
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

# Model evaluation
logger.info("Evaluating the model...")
model.eval()
y_pred_list = []
y_true_list = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, y_pred = torch.max(outputs, 1)
        y_pred_list.extend(y_pred.cpu().numpy())
        y_true_list.extend(y_batch.cpu().numpy())

# Confusion matrix
cm = confusion_matrix(y_true_list, y_pred_list)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Print classification report for more details
logger.info("Classification Report:\n" + classification_report(y_true_list, y_pred_list))

logger.info("Script completed successfully.")
