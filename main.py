import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Configuration
TEST_MODE = False  # Toggle for testing with limited data
TRAINING_DATA_PATH = r'C:\Users\alcui\Desktop\MSCE\Modules\Afstuderen\trainingdata\training.csv'  # Path to CSV file
MODEL_PATH = 'lstm_model.pth'  # Path to save the model
TARGET_COLUMN = 'malicious'  # Target variable for classification
BATCH_SIZE = 32  # Batch size for training
EPOCHS = 10 if not TEST_MODE else 2  # Reduced epochs in test mode
LEARNING_RATE = 0.0001  # Learning rate
HIDDEN_SIZE = 64  # LSTM hidden layer size
DROPOUT_RATE = 0.2  # Dropout rate for regularization
RANDOM_STATE = 42  # Random seed
SEQUENCE_LENGTH = 10  # Number of time steps per input sequence
WEIGHT_DECAY = 1e-5  # L2 regularization weight decay due to limited scope of time series
OUTPUT_SIZE = 2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Load and preprocess data
logger.info("Loading data...")
data = pd.read_csv(
    TRAINING_DATA_PATH,
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
data = data.sort_values(by='date', ascending=True)

# Some data transformations from str to labels that are usable
le = LabelEncoder()
data['method_encoded'] = le.fit_transform(data['request_method'].fillna('UNKNOWN'))
data['protocol_encoded'] = le.fit_transform(data['protocol'].fillna('UNKNOWN'))
data['referrer_encoded'] = le.fit_transform(data['referrer'].fillna('UNKNOWN'))
data['request_content_type_encoded'] = le.fit_transform(data['request_content_type'].fillna('UNKNOWN'))
data['user_agent_encoded'] = le.fit_transform(data['user_agent'].fillna('UNKNOWN'))  # Add user_agent encoding
data['request_uri_encoded'] = le.fit_transform(data['request_uri'].fillna('UNKNOWN'))  # Add user_agent encoding
data['upstream_response_time_encoded'] = le.fit_transform(data['upstream_response_time'])  # Add user_agent encoding

# Convert the IP address to integers
data['remote_ip_int'] = data['remote_ip'].apply(lambda ip: int(ip.replace('.', '')))

# Drop original columns if no longer needed
data = data.drop(columns=[
    'date', #format cannot be used
    'remote_ip', #transformed to int
    'request_method', #encoded
    'protocol', #encoded
    'referrer', #encoded
    'user_agent', #encoded
    'request_uri', #encoded
    'upstream_response_time', #too much of the same data problematic for training
    'request_content_type', #encoded
])

# Define features and target
feature_columns = [col for col in data.columns if col != TARGET_COLUMN]
x = data[feature_columns].values
y = data[TARGET_COLUMN].values

#debug
#print(data[feature_columns].dtypes)

# Convert to sequences using numpy.array() to avoid slow list-to-tensor conversion
def create_sequences(sequence_x, sequence_y, sequence_length):
    sequences = np.array([sequence_x[i:i + sequence_length] for i in range(len(sequence_x) - sequence_length + 1)])
    labels = np.array([sequence_y[i + sequence_length - 1] for i in range(len(sequence_y) - sequence_length + 1)])
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# Prepare sequential data
X_seq, Y_seq = create_sequences(x, y, SEQUENCE_LENGTH)

# Create DataLoader for training data
train_dataset = TensorDataset(X_seq, Y_seq)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
input_size = X_seq.shape[2]
model = LSTMModel(input_size, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, dropout_rate=DROPOUT_RATE).to(device)

# Criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Training loop
logger.info("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        y_batch = y_batch.to(torch.long)
        X_batch = torch.log1p(X_batch)  # brings extreme values back to normal values so the model does not get unstable.
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    logger.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss / len(train_loader):.4f}")

# Save the model after training
torch.save(model.state_dict(), MODEL_PATH)
logger.info(f"Model saved to {MODEL_PATH}")

logger.info("Script completed successfully.")
