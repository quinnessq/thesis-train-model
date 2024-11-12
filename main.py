import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# Configuration
TEST_MODE = False  # Toggle for testing with limited data
DATA_PATH = r'C:\Users\alcui\Desktop\MSCE\Modules\Afstuderen\trainingdata\training.csv'  # Path to CSV file
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
data = pd.read_csv(
    DATA_PATH,
    quotechar='"',
    sep=",",
    encoding='utf-8',
    low_memory=False,
    parse_dates=['date'],  # Parse date column
    dtype={
        'time': float,
        'malicious': bool,
        'remote_ip': str,
        'remote_port': int,
        'connection_id': int,
        'connection_time': float,
        'upstream_response_time': float,
        'upstream_response_length': int,
        'upstream_status': int,
        'upstream_connection_time': float,
        'response_body_size': int,
        'response_total_size': int,
        'response_status': int,
        'response_time': float,
        'requestLength': int,
        'request_content_length': int,
        'request_content_type': str,
        'request_method': str,
        'request_uri': str,
        'referrer': str,
        'protocol': str,
        'user_agent': str,
    }
)
# Enable test mode with limited data if TEST_MODE is True
if TEST_MODE:
    data = data.sample(n=100, random_state=RANDOM_STATE)  # Limit data to 100 samples
    logger.info("TEST_MODE is ON. Using a limited data subset.")

# Sort by time to preserve chronological order
data = data.sort_values(by='time')

# Some data transformations from str to labels that are usable
le = LabelEncoder()
data['method_encoded'] = le.fit_transform(data['request_method'])
data['protocol_encoded'] = le.fit_transform(data['protocol'])
data['referrer_encoded'] = le.fit_transform(data['referrer'])
data['request_content_type_encoded'] = le.fit_transform(data['request_content_type'])
data['remote_ip_int'] =  data['remote_ip'].apply(lambda x: int(x.replace('.', '')))
data['user_agent_encoded'] = le.fit_transform(data['user_agent'])  # Add user_agent encoding
data['request_uri_encoded'] = le.fit_transform(data['request_uri'])  # Add user_agent encoding

# Drop original columns if no longer needed
data = data.drop(columns=['date', 'request_method', 'protocol', 'referrer', 'user_agent', 'request_uri', 'remote_ip', 'request_content_type'])

# Define features and target
feature_columns = [col for col in data.columns if col != TARGET_COLUMN]
x = data[feature_columns].values
y = data[TARGET_COLUMN].values

print(data[feature_columns].dtypes)

# Convert to sequences using numpy.array() to avoid slow list-to-tensor conversion
def create_sequences(sequence_x, sequence_y, sequence_length):
    sequences = np.array([sequence_x[i:i + sequence_length] for i in range(len(sequence_x) - sequence_length + 1)])
    labels = np.array([sequence_y[i + sequence_length - 1] for i in range(len(sequence_y) - sequence_length + 1)])
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# Prepare sequential data
X_seq, y_seq = create_sequences(x, y, SEQUENCE_LENGTH)

# Split data for training and testing
split_index = int((1 - TEST_SIZE) * len(X_seq))
X_train, X_test = X_seq[:split_index], X_seq[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the LSTM model without embeddings
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM forward pass
        _, (hn, _) = self.lstm(x)  # Get the last hidden state from LSTM
        hn = self.dropout(hn[-1])  # Apply dropout

        # Fully connected layer
        out = self.fc(hn)
        return out

# Model initialization
input_size = X_train.shape[2]
model = LSTMModel(input_size, HIDDEN_SIZE, output_size=2, dropout_rate=DROPOUT_RATE).to(device)

# Criterion and optimizer
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
