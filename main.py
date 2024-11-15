import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

# Configuration
TEST_MODE = False
#TRAINING_DATA_PATH = r'C:\Users\alcui\Desktop\MSCE\Modules\Afstuderen\trainingdata\training-test.csv'
TRAINING_DATA_PATH = r'C:\Users\alcui\Desktop\MSCE\Modules\Afstuderen\trainingdata\training.csv'
#TRAINING_DATA_PATH = r'C:\Users\alcui\Desktop\MSCE\Modules\Afstuderen\trainingdata\validation.csv'
MODEL_PATH = 'lstm_model.pth'
TARGET_COLUMN = 'malicious'
BATCH_SIZE = 768
CHUNK_SIZE = 1024  # Load data in chunks
EPOCHS = 10 if not TEST_MODE else 2
LEARNING_RATE = 0.00001
HIDDEN_SIZE = 128
DROPOUT_RATE = 0.2
RANDOM_STATE = 42
SEQUENCE_LENGTH = 348
WEIGHT_DECAY = 1e-4
OUTPUT_SIZE = 2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Updated batch tokenization and padding function to ensure consistent output
def batch_tokenize_and_pad(data, column_name, max_length=15):
    # Tokenize the column with error handling for non-string or empty values
    encoded_batch = tokenizer(data[column_name].fillna('').astype(str).tolist(),
                              padding='max_length', truncation=True, max_length=max_length,
                              return_tensors="pt")
    # Convert to list of lists for DataFrame compatibility
    return encoded_batch['input_ids'].numpy().tolist()

# Updated expand_tokens function to handle non-list entries gracefully
def expand_tokens(data, column_name, num_columns):
    # Ensure each row has the correct number of tokens, replacing any non-list entries with a placeholder list
    token_data = [
        row if isinstance(row, list) and len(row) == num_columns else [0] * num_columns
        for row in data[column_name]
    ]
    # Create a DataFrame with the expanded columns
    token_columns = pd.DataFrame(token_data, columns=[f"{column_name}_{i}" for i in range(num_columns)])
    # Concatenate to the original DataFrame and drop the original token column
    data = pd.concat([data, token_columns], axis=1).drop(columns=[column_name])
    return data

# Process data in chunks
processed_chunks = []
chunk_counter = 0  # Initialize chunk counter
for chunk in pd.read_csv(
        TRAINING_DATA_PATH,
        quotechar='"',
        sep=",",
        encoding='utf-8',
        low_memory=False,
        parse_dates=['date'],
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
        },
        chunksize=CHUNK_SIZE):

    # Fill NaNs and convert non-string entries to strings in relevant columns
    chunk['request_uri'] = chunk['request_uri'].fillna('').astype(str)
    chunk['user_agent'] = chunk['user_agent'].fillna('').astype(str)

    # Tokenize and pad 'request_uri' and 'user_agent'
    chunk['request_uri_tokens_padded'] = list(batch_tokenize_and_pad(chunk, 'request_uri'))
    chunk['user_agent_tokens_padded'] = list(batch_tokenize_and_pad(chunk, 'user_agent'))

    # Expand token columns to fixed length
    chunk = expand_tokens(chunk, 'request_uri_tokens_padded', 15)
    chunk = expand_tokens(chunk, 'user_agent_tokens_padded', 15)

    # Encode categorical features
    le = LabelEncoder()
    chunk['method_encoded'] = le.fit_transform(chunk['request_method'].fillna('UNKNOWN'))
    chunk['protocol_encoded'] = le.fit_transform(chunk['protocol'].fillna('UNKNOWN'))
    chunk['referrer_encoded'] = le.fit_transform(chunk['referrer'].fillna('UNKNOWN'))
    chunk['request_content_type_encoded'] = le.fit_transform(chunk['request_content_type'].fillna('UNKNOWN'))

    # Convert IP address to integers
    chunk['remote_ip_int'] = chunk['remote_ip'].apply(lambda ip: int(ip.replace('.', '')) if pd.notna(ip) else 0)
    chunk['remote_port'] = chunk['remote_port'].fillna(-1)
    chunk['connection_id'] = chunk['connection_id'].fillna(-1)
    chunk['upstream_status'] = chunk['upstream_status'].fillna(-1)
    chunk['response_body_size'] = chunk['response_body_size'].fillna(-1)
    chunk['upstream_response_length'] = chunk['upstream_response_length'].fillna(-1)
    chunk['response_total_size'] = chunk['response_total_size'].fillna(-1)
    chunk['response_status'] = chunk['response_status'].fillna(-1)
    chunk['requestLength'] = chunk['requestLength'].fillna(-1)
    chunk['request_content_length'] = chunk['request_content_length'].fillna(-1)

    # Drop original columns
    chunk = chunk.drop(columns=[
        'date', 'remote_ip', 'request_method', 'protocol', 'referrer',
        'user_agent', 'request_uri', 'upstream_response_time', 'request_content_type',
    ])

    # Append processed chunk to list
    processed_chunks.append(chunk)
    # Increment and log chunk count
    chunk_counter += 1
    if chunk_counter % 100 == 0:  # Log every 100 batches
        logger.info(f"Processed chunk {chunk_counter}")


# Concatenate all processed chunks
data = pd.concat(processed_chunks, ignore_index=True)
logger.info(f"Total chunks processed: {chunk_counter}")

# Define features and target
feature_columns = [col for col in data.columns if col != TARGET_COLUMN]
x = data[feature_columns].values
y = data[TARGET_COLUMN].values

# Batch sequence creation
def create_sequences_batch(sequence_x, sequence_y, sequence_length, batch_size):
    """
    Generates sequences in batches to avoid memory overflow.
    """
    num_batches = len(sequence_x) // batch_size
    remainder = len(sequence_x) % batch_size

    for i in range(num_batches + (1 if remainder > 0 else 0)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(sequence_x))

        # Create sequences for the current batch
        batch_x = sequence_x[start_idx:end_idx]
        batch_y = sequence_y[start_idx:end_idx]

        # If the batch is too small to form a sequence, skip it
        if len(batch_x) < sequence_length:
            continue

        # Create sequence batches
        sequences = np.array([batch_x[j:j + sequence_length] for j in range(len(batch_x) - sequence_length + 1)])
        labels = np.array([batch_y[j + sequence_length - 1] for j in range(len(batch_y) - sequence_length + 1)])

        yield torch.tensor(sequences, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# Create DataLoader for training data using the generator
def data_generator():
    for X_batch, Y_batch in create_sequences_batch(x, y, SEQUENCE_LENGTH, BATCH_SIZE):

        # Ensure X_batch contains valid numbers before logging and transforming
        #print(f"Original unique values in Y_batch: {torch.unique(Y_batch)}")
        #print(f"Original unique values in X_batch: {torch.unique(X_batch)}")

        # Clean X_batch by handling NaN, positive infinity, and negative infinity
        X_batch = torch.nan_to_num(X_batch, nan=0.0, posinf=1e10, neginf=-1e10)

        # Check if there are still any NaN or infinite values
        if torch.any(torch.isnan(X_batch)) or torch.any(torch.isinf(X_batch)):
            print("Warning: X_batch contains NaN or Inf values after cleaning")
            X_batch = torch.nan_to_num(X_batch, nan=0.0, posinf=1e10, neginf=-1e10)

        # Print unique values after cleaning X_batch
        #print(f"Unique values in cleaned X_batch: {torch.unique(X_batch)}")

        # Ensure that X_batch doesn't contain negative values before applying log1p
        X_batch = torch.clamp(X_batch, min=0.0)  # Clip values to >= 0

        # Apply log1p to avoid instability due to large values
        X_batch = torch.log1p(X_batch).to(device)  # Apply log1p and send to device

        # Clean Y_batch to ensure no invalid values
        invalid_label_value = -9223372036854775808  # Invalid label (int64 min value)
        valid_class_range = 10  # Assuming 10 classes, adjust based on your specific task

        # Replace invalid labels in Y_batch with a valid class (e.g., 0)
        Y_batch = torch.where(Y_batch == invalid_label_value, torch.zeros_like(Y_batch), Y_batch)

        # Ensure Y_batch values are within the valid range [0, n_classes)
        Y_batch = torch.clamp(Y_batch, min=0, max=valid_class_range - 1)

        # Check for any invalid Y_batch values
        if torch.any(Y_batch < 0) or torch.any(Y_batch >= valid_class_range):
            print(f"Warning: Y_batch contains invalid values: {torch.unique(Y_batch)}")

        # Ensure Y_batch is transferred to the device
        Y_batch = Y_batch.to(device)

        # Print unique values of Y_batch after cleaning
        #print(f"Unique values in cleaned Y_batch: {torch.unique(Y_batch)}")

        # Yield the batch
        yield X_batch, Y_batch



# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=4, bidirectional=True, batch_first=True, dropout=dropout_rate)
        self.attention = nn.Linear(hidden_size * 2, 1)  # Attention layer (if bidirectional, hidden_size * 2)

        # Additional fully connected layers with dropout and layer normalization
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # LSTM output
        # Apply attention mechanism
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # Apply attention weights

        # Fully connected layers with layer normalization and dropout
        out = self.fc1(context)
        out = self.relu(out)
        out = self.layer_norm(out)  # Layer normalization
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        return out


# Model initialization
input_size = len(feature_columns)
logger.info(f"Feature column size: {input_size}")
model = LSTMModel(input_size, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, dropout_rate=DROPOUT_RATE).to(device)

# Criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training loop
logger.info("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    batch_counter = 0  # To count the number of batches processed

    for X_batch, y_batch in data_generator():
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        # Update learning rate

        running_loss += loss.item()
        batch_counter += 1

        #if batch_counter % 100 == 0:  # Log every 100 batches
            #logger.info(f"Processed {batch_counter} batches in epoch {epoch+1}")

    scheduler.step()
    logger.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss / batch_counter:.4f}")

# Save the model after training
torch.save(model.state_dict(), MODEL_PATH)
logger.info(f"Model saved to {MODEL_PATH}")

logger.info("Script completed successfully.")
