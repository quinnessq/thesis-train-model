import logging
import os
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
PROCESSED_DATA_PATH = r'C:\Users\alcui\Desktop\MSCE\Modules\Afstuderen\trainingdata\processed_data_training.pkl'  # Path for processed data
#PROCESSED_DATA_PATH = r'C:\Users\alcui\Desktop\MSCE\Modules\Afstuderen\trainingdata\processed_data_training_test.pkl'  # Path for processed data

VALIDATION_DATA_PATH = r'C:\Users\alcui\Desktop\MSCE\Modules\Afstuderen\trainingdata\validation.csv'  # Path to CSV file
PROCESSED_DATA_VALIDATION_PATH = r'C:\Users\alcui\Desktop\MSCE\Modules\Afstuderen\trainingdata\processed_data_validation.pkl'  # Path for processed data

MODEL_PATH = 'lstm_model.pth'
TARGET_COLUMN = 'malicious'
BATCH_SIZE = 128
CHUNK_SIZE = 512
HIDDEN_SIZE = 64
SEQUENCE_LENGTH = 64

EPOCHS = 10 if not TEST_MODE else 2
LEARNING_RATE = 0.00001
WEIGHT_DECAY = 0.02
DROPOUT_RATE = 0.2
OUTPUT_SIZE = 2

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
    token_data = [
        row if isinstance(row, list) and len(row) == num_columns else [0] * num_columns
        for row in data[column_name]
    ]
    token_columns = pd.DataFrame(token_data, columns=[f"{column_name}_{i}" for i in range(num_columns)])
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

    # Replace NaNs with empty strings and convert to strings using .loc
    chunk.loc[:, 'request_uri'] = chunk['request_uri'].fillna('').astype(str)
    chunk.loc[:, 'user_agent'] = chunk['user_agent'].fillna('').astype(str)

    # Tokenization and padding
    chunk['request_uri_tokens_padded'] = batch_tokenize_and_pad(chunk, 'request_uri')
    chunk['user_agent_tokens_padded'] = batch_tokenize_and_pad(chunk, 'user_agent')

    # Expanding tokens into fixed number of columns
    chunk = expand_tokens(chunk, 'request_uri_tokens_padded', 15)
    chunk = expand_tokens(chunk, 'user_agent_tokens_padded', 15)

    # Encode categorical features
    le = LabelEncoder()
    chunk.loc[:, 'method_encoded'] = le.fit_transform(chunk['request_method'].fillna('UNKNOWN'))
    chunk.loc[:, 'protocol_encoded'] = le.fit_transform(chunk['protocol'].fillna('UNKNOWN'))
    chunk.loc[:, 'referrer_encoded'] = le.fit_transform(chunk['referrer'].fillna('UNKNOWN'))
    chunk.loc[:, 'request_content_type_encoded'] = le.fit_transform(chunk['request_content_type'].fillna('UNKNOWN'))

    # Convert IP addresses and handle missing values
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

# Check if processed data exists and load it, otherwise process raw data
if os.path.exists(PROCESSED_DATA_PATH):
    logger.info("Loading processed data from file...")
    data = pd.read_pickle(PROCESSED_DATA_PATH)
else:
    logger.info("Loading and sorting the dataset...")
    data = pd.read_csv(TRAINING_DATA_PATH, sep=",", encoding='utf-8', low_memory=False)

    # Sort by connection_id and time to ensure the data is in correct order
    data.sort_values(by=['connection_id', 'time'], inplace=True)

    # Calculate time difference between requests for each connection_id (in seconds)
    data['time_diff'] = data.groupby('connection_id')['time'].diff()

    # Calculate session duration for each connection_id (in seconds)
    data['session_duration'] = data.groupby('connection_id')['time'].transform(lambda x: x.max() - x.min())

    # Process the data in chunks
    processed_chunks = []
    chunk_counter = 0  # Initialize chunk counter
    for start_row in range(0, len(data), CHUNK_SIZE):
        # Create a chunk
        chunk = data.iloc[start_row:start_row+CHUNK_SIZE]

        # Process the chunk
        processed_chunks.append(process_chunk(chunk))
        chunk_counter += 1
        if chunk_counter % 100 == 0:  # Log every 100 chunks
            logger.info(f"Processed chunk {chunk_counter}")

    # Concatenate all processed chunks
    data = pd.concat(processed_chunks, ignore_index=True)

    # Save processed data for future use
    logger.info(f"Saving processed data to {PROCESSED_DATA_PATH}...")
    data.to_pickle(PROCESSED_DATA_PATH)

# Define features and target
feature_columns = [col for col in data.columns if col != TARGET_COLUMN]
x = data[feature_columns].values
y = data[TARGET_COLUMN].values

def create_sequences_batch(sequence_x, sequence_y, sequence_length, batch_size):
    """ Generate sequences in batches to avoid memory overflow """
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
    """ Generator function to yield batches of data """
    for X_batch, Y_batch in create_sequences_batch(x, y, SEQUENCE_LENGTH, BATCH_SIZE):
        X_batch = torch.nan_to_num(X_batch, nan=0.0, posinf=1e10, neginf=-1e10)
        X_batch = torch.clamp(X_batch, min=0.0)
        X_batch = torch.log1p(X_batch).to(device)

        # Clean labels and ensure they're within valid class range
        Y_batch = torch.clamp(Y_batch, min=0, max=OUTPUT_SIZE - 1).to(device)
        yield X_batch, Y_batch

def process_validation_data():
    """ Processes the validation data similarly to the training data. """
    # Load validation data
    logger.info("Loading validation data...")
    data = pd.read_csv(VALIDATION_DATA_PATH, sep=",", encoding='utf-8', low_memory=False)

    # Sort by connection_id and time to ensure the data is in correct order
    data.sort_values(by=['connection_id', 'time'], inplace=True)

    # Calculate time difference between requests for each connection_id (in seconds)
    data['time_diff'] = data.groupby('connection_id')['time'].diff()

    # Calculate session duration for each connection_id (in seconds)
    data['session_duration'] = data.groupby('connection_id')['time'].transform(lambda x: x.max() - x.min())

    # Process the data in chunks
    processed_chunks = []
    chunk_counter = 0
    for start_row in range(0, len(data), CHUNK_SIZE):
        chunk = data.iloc[start_row:start_row + CHUNK_SIZE]
        processed_chunks.append(process_chunk(chunk))
        chunk_counter += 1
        if chunk_counter % 100 == 0:  # Log every 100 chunks
            logger.info(f"Processed chunk {chunk_counter}")

    # Concatenate processed chunks
    data = pd.concat(processed_chunks, ignore_index=True)

    # Save processed data
    logger.info(f"Saving processed validation data to {PROCESSED_DATA_PATH}...")
    data.to_pickle(PROCESSED_DATA_PATH)

    logger.info("Validation data processing complete.")
    return data

# Load or process validation data
if os.path.exists(PROCESSED_DATA_VALIDATION_PATH):
    logger.info("Loading processed validation data from file...")
    validation_data = pd.read_pickle(PROCESSED_DATA_VALIDATION_PATH)
else:
    validation_data = process_validation_data()

# Define features and target for validation data
validation_feature_columns = [col for col in validation_data.columns if col != TARGET_COLUMN]
validation_x = validation_data[validation_feature_columns].values
validation_y = validation_data[TARGET_COLUMN].values

# Function to create sequences for validation data in batches
def validation_data_generator():
    """ Generator function for validation data in batches """
    for X_batch, Y_batch in create_sequences_batch(validation_x, validation_y, SEQUENCE_LENGTH, BATCH_SIZE):
        X_batch = torch.nan_to_num(X_batch, nan=0.0, posinf=1e10, neginf=-1e10)
        X_batch = torch.clamp(X_batch, min=0.0)
        X_batch = torch.log1p(X_batch).to(device)

        # Clean labels and ensure they're within valid class range
        Y_batch = torch.clamp(Y_batch, min=0, max=OUTPUT_SIZE - 1).to(device)
        yield X_batch, Y_batch

logger.info("Data processing complete.")

# Define the LSTM model with multi-head attention
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(LSTMModel, self).__init__()

        # LSTM layer (bidirectional)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=4, bidirectional=True, batch_first=True, dropout=dropout_rate)

        # Transformer-based multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=4, dropout=dropout_rate)

        # Fully connected layers (MLP)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # ReLU activation
        self.relu = nn.ReLU()

        # Layer normalization (after dropout and activation)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Get LSTM outputs
        lstm_out, _ = self.lstm(x)

        # Transformer-based attention
        lstm_out_transpose = lstm_out.permute(1, 0, 2)  # (batch, seq_len, hidden_size*2) -> (seq_len, batch, hidden_size*2)

        # Pass through multihead attention layer
        attn_output, _ = self.attention(lstm_out_transpose, lstm_out_transpose, lstm_out_transpose)

        # The output is the weighted sum of the input sequence, now we revert the permutation
        attn_output = attn_output.permute(1, 0, 2)  # (seq_len, batch, hidden_size*2) -> (batch, seq_len, hidden_size*2)

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

        out = self.fc3(out)  # Final output (raw logits)

        return out

# Model initialization
model = LSTMModel(input_size=len(feature_columns), hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, dropout_rate=DROPOUT_RATE).to(device)

# Criterion and optimizer
class_weights = torch.tensor([1.0, 10.0])  # Heavier penalty for class 1
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# Early stopping parameters
early_stopping_patience = 2  # Stop after 5 epochs without improvement
best_val_loss = float('inf')  # Initialize to a very high value
epochs_without_improvement = 0

logger.info("Start training model...")

# Training loop
for epoch in range(EPOCHS):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    batch_counter = 0

    # Training loop
    for X_batch, y_batch in data_generator():
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        batch_counter += 1
        if batch_counter % 100 == 0:
            logger.info(f"Epoch {epoch+1}/{EPOCHS}, Processed {batch_counter} batches, Running Loss: {running_loss / batch_counter:.4f}")

    # Log training loss at the end of the epoch
    logger.info(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {running_loss / batch_counter:.4f}")

    # Validation loop: calculate validation loss once per epoch
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    val_counter = 0
    with torch.no_grad():
        for X_batch_val, y_batch_val in validation_data_generator():
            outputs_val = model(X_batch_val)
            loss_val = criterion(outputs_val, y_batch_val)
            val_loss += loss_val.item()
            val_counter += 1

    # Log validation loss at the end of the epoch
    avg_val_loss = val_loss / val_counter
    logger.info(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {avg_val_loss:.4f}")

    # Step scheduler: reduce learning rate if validation loss plateaus
    scheduler.step(avg_val_loss)

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0  # Reset counter if we have an improvement
    else:
        epochs_without_improvement += 1
        logger.info(f"No improvement for {epochs_without_improvement} epochs.")

    # Stop training if validation loss hasn't improved for `early_stopping_patience` epochs
    if epochs_without_improvement >= early_stopping_patience:
        logger.info(f"Validation loss has not improved for {early_stopping_patience} epochs. Stopping training.")
        break

# Save the model at the end of training
torch.save(model.state_dict(), MODEL_PATH)
logger.info(f"Model saved to {MODEL_PATH}")
