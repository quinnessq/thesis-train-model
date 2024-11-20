import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer

# Configuration
TEST_MODE = False
TRAINING_DATA_PATH = r'C:\Users\alcui\Desktop\MSCE\Modules\Afstuderen\trainingdata\training.csv'
PROCESSED_DATA_PATH = r'C:\Users\alcui\Desktop\MSCE\Modules\Afstuderen\trainingdata\processed_data_training.pkl'  # Path for processed data

VALIDATION_DATA_PATH = r'C:\Users\alcui\Desktop\MSCE\Modules\Afstuderen\trainingdata\validation.csv'  # Path to CSV file
PROCESSED_DATA_VALIDATION_PATH = r'C:\Users\alcui\Desktop\MSCE\Modules\Afstuderen\trainingdata\processed_data_validation.pkl'  # Path for processed data

MODEL_PATH = 'lstm_model.pth'
TARGET_COLUMN = 'malicious'
BATCH_SIZE = 8192
HIDDEN_SIZE = 64
SEQUENCE_LENGTH = 8192

CHUNK_SIZE = 2000
EPOCHS = 30 if not TEST_MODE else 1
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.05
DROPOUT_RATE = 0.3
OUTPUT_SIZE = 2
EARLY_STOPPING_PATIENCE = 10

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def batch_tokenize_and_pad(data, column_name, max_length=15):
    """ Tokenize and pad a given column of data to max_length """
    encoded_batch = tokenizer(data[column_name].fillna('').astype(str).tolist(),
                              padding='max_length', truncation=True, max_length=max_length,
                              return_tensors="pt")
    return encoded_batch['input_ids'].numpy().tolist()

def expand_tokens(data, column_name, num_columns):
    """ Expand tokens into a fixed number of columns """

    # Tokenize each entry and ensure it's a list of exact length `num_columns`
    token_data = [
        (row if isinstance(row, list) and len(row) == num_columns
         else [0] * num_columns)  # Pad with zeros if row is not a valid list or doesn't match `num_columns`
        for row in data[column_name].fillna('')  # Ensure NaNs are handled by filling with empty string
    ]

    # Create new column names and check for conflicts
    new_column_names = [f"{column_name}_{i}" for i in range(num_columns)]

    # If any of the new column names already exist, we append a suffix to make them unique
    existing_columns = set(data.columns)
    for i, new_col in enumerate(new_column_names):
        if new_col in existing_columns:
            new_column_names[i] = f"{new_col}_new"

    # Create the token columns DataFrame
    token_columns = pd.DataFrame(token_data, columns=new_column_names)

    # Reset index for both data and token_columns to align rows properly during concatenation
    data.reset_index(drop=True, inplace=True)
    token_columns.reset_index(drop=True, inplace=True)

    # Concatenate the new token columns and drop the original column
    data = pd.concat([data, token_columns], axis=1).drop(columns=[column_name])

    return data

def aggregate_connection_features(data):
    """ Aggregate features for each connection_id """

    # Aggregate features
    aggregated_features = data.groupby('connection_id').agg(
        num_requests=('request_uri', 'count'),
        avg_response_time=('response_time', 'mean'),
        max_response_time=('response_time', 'max'),
        count_slow_requests=('response_time', lambda x: (x > 0.75).sum()),  # Slow requests > 0.75s
        avg_time_between_requests=('time_diff', 'mean'),  # Average time between requests
        session_duration=('session_duration', 'mean'),  # Duration of session (mean)
        total_response_size=('response_body_size', 'sum'),
        avg_response_size=('response_body_size', 'mean'),
        unique_uris=('request_uri', 'nunique'),  # Count unique request URIs
    ).reset_index()

    # Now calculate request_frequency as requests per unit time (e.g., per second)
    aggregated_features['request_frequency'] = aggregated_features['num_requests'] / aggregated_features['session_duration']

    return aggregated_features

def process_chunk(chunk):
    """ Process each chunk of data """
    # Ensure chunk is a copy to avoid SettingWithCopyWarning
    chunk = chunk.copy()

    chunk.loc[:, 'request_uri'] = chunk['request_uri'].fillna('').astype(str)
    chunk.loc[:, 'user_agent'] = chunk['user_agent'].fillna('').astype(str)

    chunk['request_uri_tokens_padded'] = batch_tokenize_and_pad(chunk, 'request_uri')
    chunk['user_agent_tokens_padded'] = batch_tokenize_and_pad(chunk, 'user_agent')

    chunk = expand_tokens(chunk, 'request_uri_tokens_padded', 15)
    chunk = expand_tokens(chunk, 'user_agent_tokens_padded', 15)

    le = LabelEncoder()
    chunk.loc[:, 'method_encoded'] = le.fit_transform(chunk['request_method'].fillna('UNKNOWN'))
    chunk.loc[:, 'protocol_encoded'] = le.fit_transform(chunk['protocol'].fillna('UNKNOWN'))
    chunk.loc[:, 'referrer_encoded'] = le.fit_transform(chunk['referrer'].fillna('UNKNOWN'))
    chunk.loc[:, 'request_content_type_encoded'] = le.fit_transform(chunk['request_content_type'].fillna('UNKNOWN'))

    chunk.loc[:, 'remote_ip_int'] = chunk['remote_ip'].apply(lambda ip: int(ip.replace('.', '')) if pd.notna(ip) else 0)
    chunk.fillna({
        'remote_port': -1,
        'connection_id': -1,
        'upstream_status': -1,
        'response_body_size': -1,
        'upstream_response_length': -1,
        'response_total_size': -1,
        'response_status': -1,
        'requestLength': -1,
        'request_content_length': -1
    }, inplace=True)

    # Create aggregated connection features
    aggregated_features = aggregate_connection_features(chunk)

    chunk = chunk.merge(aggregated_features, on='connection_id', how='left')

    # Drop original columns we don't need anymore
    chunk.drop(columns=['date', 'remote_ip', 'request_method', 'protocol', 'referrer', 'user_agent', 'request_uri',
                        'upstream_response_time', 'request_content_type'], inplace=True)
    return chunk

def process_training_data():
    data = pd.read_csv(TRAINING_DATA_PATH, sep=",", encoding='utf-8', low_memory=False)
    logger.info(f"Data type of 'malicious' before conversion: {data['malicious'].dtype}")
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype('int')  # Convert to integer
    logger.info(f"Data type of 'malicious' after conversion: {data['malicious'].dtype}")
    print(f"After changing data type for all data: {np.unique(data[TARGET_COLUMN])}")

    groups = [df for _, df in data.groupby('connection_id')]
    random.shuffle(groups)
    data = pd.concat(groups, ignore_index=True)
    logger.info("Randomized the order of connection_id groups and sorted each group by time.")

    data['time_diff'] = data.groupby('connection_id')['time'].diff()
    data['session_duration'] = data.groupby('connection_id')['time'].transform(lambda x: x.max() - x.min())

    # Process the data in chunks
    processed_chunks = []
    chunk_counter = 0  # Initialize chunk counter
    previous_chunk = None
    for start_row in range(0, len(data), CHUNK_SIZE):
        # Ensure we do not exceed the bounds of the DataFrame
        end_row = min(start_row + CHUNK_SIZE, len(data))
        chunk = data.iloc[start_row:end_row]

        # Print unique values of the target column for the current chunk
        unique_values = np.unique(chunk[TARGET_COLUMN])
        if 1 in unique_values:
            print(f"Chunk {chunk_counter} contains malicious entries (1): {unique_values}")

        if previous_chunk is not None:
            if chunk.equals(previous_chunk):
                logger.warning(f"Chunk {chunk_counter} has identical rows as the previous chunk.")
        previous_chunk = chunk.copy()

        # Process the chunk
        processed_chunks.append(process_chunk(chunk))
        chunk_counter += 1

        # Log every 100 chunks
        if chunk_counter % 100 == 0:
            logger.info(f"Processed chunk {chunk_counter}")
    # Concatenate all processed chunks
    data = pd.concat(processed_chunks, ignore_index=True)
    # Save processed data for future use
    logger.info(f"Saving processed data to {PROCESSED_DATA_PATH}...")
    data.to_pickle(PROCESSED_DATA_PATH)

    return data

def process_validation_data():
    """ Processes the validation data similarly to the training data. """
    # Load validation data
    logger.info("Loading validation data...")
    data = pd.read_csv(VALIDATION_DATA_PATH, sep=",", encoding='utf-8', low_memory=False)

    logger.info(f"Data type of 'malicious' before conversion: {data['malicious'].dtype}")
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype('int')  # Convert to integer
    logger.info(f"Data type of 'malicious' after conversion: {data['malicious'].dtype}")
    print(f"After changing data type for all data: {np.unique(data[TARGET_COLUMN])}")

    # Sort by connection_id and time to ensure the data is in correct order
    groups = [df for _, df in data.groupby('connection_id')]
    random.shuffle(groups)
    data = pd.concat(groups, ignore_index=True)
    logger.info("Randomized the order of connection_id groups.")

    data['time_diff'] = data.groupby('connection_id')['time'].diff()
    data['session_duration'] = data.groupby('connection_id')['time'].transform(lambda x: x.max() - x.min())

    # Process the data in chunks
    processed_chunks = []
    chunk_counter = 0
    # Initialize the previous chunk for comparison
    previous_chunk = None

    for start_row in range(0, len(data), CHUNK_SIZE):
        # Ensure we do not exceed the bounds of the DataFrame
        end_row = min(start_row + CHUNK_SIZE, len(data))
        chunk = data.iloc[start_row:end_row]

        # Print unique values of the target column for the current chunk
        unique_values = np.unique(chunk[TARGET_COLUMN])
        if 1 in unique_values:
            print(f"Chunk {chunk_counter} contains malicious entries (1): {unique_values}")

        # If this is not the first chunk, compare it with the previous one
        if previous_chunk is not None:
            # Compare entire chunks (rows) between the current and previous chunks
            if chunk.equals(previous_chunk):
                logger.warning(f"Chunk {chunk_counter} has identical rows as the previous chunk.")

        # Save the current chunk for comparison in the next iteration
        previous_chunk = chunk.copy()

        # Process the chunk
        processed_chunks.append(process_chunk(chunk))
        chunk_counter += 1

        # Log every 100 chunks
        if chunk_counter % 100 == 0:
            logger.info(f"Processed chunk {chunk_counter}")

    # Concatenate processed chunks
    data = pd.concat(processed_chunks, ignore_index=True)

    print(f"Data after processing chunks contains: {np.unique(data[TARGET_COLUMN])}")

    # Save processed data
    logger.info(f"Saving processed validation data to {PROCESSED_DATA_VALIDATION_PATH}...")
    data.to_pickle(PROCESSED_DATA_VALIDATION_PATH)

    logger.info("Validation data processing complete.")
    return data

def create_sequences_batch(sequence_x, sequence_y, sequence_length, batch_size):
    num_batches = len(sequence_x) // batch_size
    remainder = len(sequence_x) % batch_size
    for i in range(num_batches + (1 if remainder > 0 else 0)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(sequence_x))

        batch_x = sequence_x[start_idx:end_idx]
        batch_y = sequence_y[start_idx:end_idx]

        if len(batch_x) < sequence_length:
            continue

        # Create sequence batches
        sequences = np.array([batch_x[j:j + sequence_length] for j in range(len(batch_x) - sequence_length + 1)])
        labels = np.array([batch_y[j + sequence_length - 1] for j in range(len(batch_y) - sequence_length + 1)])

        sequences = torch.stack([torch.tensor(seq, dtype=torch.float32) for seq in sequences])
        labels = torch.tensor(labels, dtype=torch.long)

        yield sequences, labels

def data_generator():
    """ Generator function to yield batches of data for training """
    for X_batch, Y_batch in create_sequences_batch(x, y, SEQUENCE_LENGTH, BATCH_SIZE):
        # Ensure input data is on the correct device
        X_batch = torch.nan_to_num(X_batch, nan=0.0, posinf=1e10, neginf=-1e10)
        X_batch = torch.clamp(X_batch, min=0.0)
        X_batch = torch.log1p(X_batch).to(device)  # Move input data to device

        # Ensure labels are on the correct device
        Y_batch = torch.clamp(Y_batch, min=0, max=OUTPUT_SIZE - 1).to(device)  # Move labels to device

        # Yield the batch (input data and corresponding labels)
        yield X_batch, Y_batch

def validation_data_generator():
    """ Generator function to yield batches of validation data """
    for X_batch, Y_batch in create_sequences_batch(validation_x, validation_y, SEQUENCE_LENGTH, BATCH_SIZE):
        # Ensure input data is on the correct device
        X_batch = torch.nan_to_num(X_batch, nan=0.0, posinf=1e10, neginf=-1e10)
        X_batch = torch.clamp(X_batch, min=0.0)
        X_batch = torch.log1p(X_batch).to(device)  # Move input data to device

        # Clean labels and ensure they're within valid class range
        Y_batch = torch.clamp(Y_batch, min=0, max=OUTPUT_SIZE - 1).to(device)  # Ensure labels are on the correct device

        # Yield the batch (input data and corresponding labels)
        yield X_batch, Y_batch


# Check if processed data exists and load it, otherwise process raw data
if os.path.exists(PROCESSED_DATA_PATH):
    logger.info("Loading processed data from file...")
    data = pd.read_pickle(PROCESSED_DATA_PATH)
else:
    logger.info("Loading and sorting the dataset...")
    data = process_training_data()

# Define features and target
train_data, validation_data = train_test_split(data, test_size=0.2, random_state=42)

feature_columns = [col for col in data.columns if col != TARGET_COLUMN]
x = train_data[feature_columns].values
y = train_data[TARGET_COLUMN].values

# Define features and target for validation data
validation_feature_columns = [col for col in validation_data.columns if col != TARGET_COLUMN]
validation_x = validation_data[validation_feature_columns].values
validation_y = validation_data[TARGET_COLUMN].values

logger.info("Data processing complete.")

# Define the LSTM model with multi-head attention
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(LSTMModel, self).__init__()

        # LSTM layer (unidirectional) with an additional layer (num_layers=5)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=5, bidirectional=False, batch_first=True, dropout=dropout_rate)

        # Transformer-based multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2, dropout=dropout_rate)

        # Fully connected layers (MLP)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, output_size)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # ReLU activation
        self.relu = nn.ReLU()

        # Layer normalization (after dropout and activation)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, hidden=None):
        """
        Forward pass for LSTM model. If hidden states are provided, use them, else initialize new hidden states.

        :param x: Input tensor with shape (batch_size, sequence_length, input_size)
        :param hidden: Tuple (h0, c0) containing initial hidden and cell states. If None, they will be initialized.
        :return: The output after passing through LSTM, attention, and fully connected layers.
        """
        # If hidden states are not passed, initialize them as zeros
        if hidden is None:
            h0 = torch.zeros(5, x.size(0), self.lstm.hidden_size).to(x.device)  # num_layers=5
            c0 = torch.zeros(5, x.size(0), self.lstm.hidden_size).to(x.device)  # num_layers=5
        else:
            h0, c0 = hidden

        # Get LSTM outputs and hidden states
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # Detach hidden states from the computation graph to prevent backpropagating through previous batches
        h_n = h_n.detach()
        c_n = c_n.detach()

        # Transformer-based attention
        lstm_out_transpose = lstm_out.permute(1, 0, 2)  # (batch, seq_len, hidden_size) -> (seq_len, batch, hidden_size)

        # Pass through multihead attention layer
        attn_output, _ = self.attention(lstm_out_transpose, lstm_out_transpose, lstm_out_transpose)

        # The output is the weighted sum of the input sequence, now we revert the permutation
        attn_output = attn_output.permute(1, 0, 2)  # (seq_len, batch, hidden_size) -> (batch, seq_len, hidden_size)

        # Use the last time step's output for classification
        context = attn_output[:, -1, :]  # Take the output at the last timestep

        # Fully connected layers with ReLU activations and dropout
        out = self.fc1(context)
        out = self.relu(out)
        out = self.layer_norm(out)  # Layer normalization
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)  # Third FC layer
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc4(out)  # Final output (raw logits)

        # Return the output and the new hidden states for the next iteration
        return out, (h_n, c_n)

# Model initialization
model = LSTMModel(input_size=len(feature_columns), hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, dropout_rate=DROPOUT_RATE).to(device)

# Criterion and optimizer
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=4.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Use the computed class weights in your loss function
criterion = FocalLoss(alpha=0.35, gamma=4.0, weight=class_weights)
#criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# Early stopping parameters
best_val_loss = float('inf')
last_val_loss = float('inf')
epochs_without_improvement = 0

logger.info("Start training model...")

def train_data():
    global best_val_loss, epochs_without_improvement
    for epoch in range(EPOCHS):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        batch_counter = 0
        correct_preds = 0  # Track correct predictions for accuracy
        total_preds = 0  # Track total predictions

        # Initialize the hidden state for training (set to None for LSTM/GRU)
        hidden = None  # Reset hidden state at the start of each epoch

        # Training loop
        for X_batch, y_batch in data_generator():
            optimizer.zero_grad()

            # Pass the current hidden state to the model and get the output
            outputs, hidden = model(X_batch, hidden)

            # Calculate loss
            loss = criterion(outputs, y_batch)

            # Perform backward pass and update weights (without retain_graph)
            loss.backward()

            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # Update the model parameters
            optimizer.step()

            # Track running loss and accuracy
            running_loss += loss.item()
            batch_counter += 1

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)  # Get the predicted class (assumes classification task)
            correct_preds += (predicted == y_batch).sum().item()  # Count correct predictions
            total_preds += y_batch.size(0)  # Count total number of examples

            if batch_counter % 100 == 0:
                logger.info(f"Epoch {epoch + 1}/{EPOCHS}, Processed {batch_counter} batches, "
                            f"Running Loss: {running_loss / batch_counter:.4f}")

        # Log training loss and accuracy at the end of the epoch
        avg_train_loss = running_loss / batch_counter
        train_accuracy = correct_preds / total_preds  # Calculate training accuracy
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {avg_train_loss:.4f}, "
                    f"Training Accuracy: {train_accuracy:.4f}")

        # Validation loop: calculate validation loss and accuracy once per epoch
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_counter = 0
        val_correct_preds = 0  # Track correct predictions for validation accuracy
        val_total_preds = 0  # Track total predictions for validation

        # Initialize the hidden state for validation
        hidden = None  # Reset hidden state at the start of validation

        with torch.no_grad():
            for X_batch_val, y_batch_val in validation_data_generator():
                # Pass the current hidden state to the model and get the output
                outputs_val, hidden = model(X_batch_val, hidden)

                # Compute validation loss
                loss_val = criterion(outputs_val, y_batch_val)
                val_loss += loss_val.item()
                val_counter += 1

                # Calculate validation accuracy
                _, predicted_val = torch.max(outputs_val, 1)  # Get the predicted class
                val_correct_preds += (predicted_val == y_batch_val).sum().item()  # Count correct predictions
                val_total_preds += y_batch_val.size(0)  # Count total number of validation examples

        # Log validation loss and accuracy at the end of the epoch
        avg_val_loss = val_loss / val_counter
        val_accuracy = val_correct_preds / val_total_preds  # Calculate validation accuracy
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}, Validation Loss: {avg_val_loss:.4f}, "
                    f"Validation Accuracy: {val_accuracy:.4f}")

        # Step scheduler: reduce learning rate if validation loss plateaus
        scheduler.step(avg_val_loss)

        last_val_loss = avg_val_loss

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0  # Reset counter if we have an improvement
            torch.save(model.state_dict(), MODEL_PATH)  # Save model with best validation loss
            logger.info(f"Model saved to {MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epochs.")

        # Stop training if validation loss hasn't improved for `early_stopping_patience` epochs
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Validation loss has not improved for {EARLY_STOPPING_PATIENCE} epochs. Stopping training.")
            break

# Training loop
train_data()

# Save the model at the end of training
if last_val_loss < best_val_loss:
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")
