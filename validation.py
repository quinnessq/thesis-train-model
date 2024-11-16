import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

# Configuration
#VALIDATION_DATA_PATH = r'C:\Users\alcui\Desktop\MSCE\Modules\Afstuderen\trainingdata\training-test.csv'
VALIDATION_DATA_PATH = r'C:\Users\alcui\Desktop\MSCE\Modules\Afstuderen\trainingdata\validation.csv'  # Path to CSV file
TARGET_COLUMN = 'malicious'  # Target variable for classification
MODEL_PATH = 'lstm_model.pth'
BATCH_SIZE = 512
CHUNK_SIZE = 1024  # Load data in chunks
HIDDEN_SIZE = 256
DROPOUT_RATE = 0.2
SEQUENCE_LENGTH = 256
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

def process_chunk(chunk):
    """ Process each chunk of data """
    # Ensure chunk is a copy to avoid SettingWithCopyWarning
    chunk = chunk.copy()

    # Replace NaNs with empty strings and convert to strings
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
    chunk['method_encoded'] = le.fit_transform(chunk['request_method'].fillna('UNKNOWN'))
    chunk['protocol_encoded'] = le.fit_transform(chunk['protocol'].fillna('UNKNOWN'))
    chunk['referrer_encoded'] = le.fit_transform(chunk['referrer'].fillna('UNKNOWN'))
    chunk['request_content_type_encoded'] = le.fit_transform(chunk['request_content_type'].fillna('UNKNOWN'))

    # Convert IP addresses and handle missing values
    chunk['remote_ip_int'] = chunk['remote_ip'].apply(lambda ip: int(ip.replace('.', '')) if pd.notna(ip) else 0)
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

    # Drop original columns
    chunk.drop(columns=['date', 'remote_ip', 'request_method', 'protocol', 'referrer', 'user_agent', 'request_uri',
                        'upstream_response_time', 'request_content_type'], inplace=True)

    return chunk

# Load and sort the entire dataset before chunking
logger.info("Loading and sorting the dataset...")
data = pd.read_csv(TRAINING_DATA_PATH, sep=",", encoding='utf-8', low_memory=False)

# Sort by connection_id and time to ensure the data is in correct order
data.sort_values(by=['connection_id', 'time'], inplace=True)

# Now process the data in chunks
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

logger.info("Data processing complete.")


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

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(LSTMModel, self).__init__()

        # LSTM layer (bidirectional)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=4, bidirectional=True, batch_first=True, dropout=dropout_rate)

        # Attention layer
        self.attention = nn.Linear(hidden_size * 2, 1)  # attention weight for each timestep in LSTM

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

        # Compute attention weights and apply them to the LSTM output
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # attention weights over timesteps
        context = torch.sum(attn_weights * lstm_out, dim=1)  # weighted sum of LSTM output

        # Fully connected layers with ReLU activations and dropout
        out = self.fc1(context)
        out = self.relu(out)
        out = self.layer_norm(out)  # Layer normalization
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)  # Final output (raw logits, no activation)

        return out

# Model initialization
input_size = len(feature_columns)
logger.info(f"Feature column size: {input_size}")
# Model initialization
model = LSTMModel(input_size=len(feature_columns), hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, dropout_rate=DROPOUT_RATE).to(device)

# Load the pre-trained model if available
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH), strict=False)
    logger.info("Loaded existing model from disk.")
else:
    logger.error("No model found at the specified path.")
    exit()

# Model evaluation
logger.info("Evaluating the model...")

# Initialize the loss function once
criterion = nn.CrossEntropyLoss()

model.eval()
y_pred_list = []
y_true_list = []

# Disable gradient computation for validation
with torch.no_grad():
    for X_batch, y_batch in data_generator():
        # Move data to the correct device (GPU or CPU)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Get model predictions
        outputs = model(X_batch)

        # Compute loss (optional)
        loss = criterion(outputs, y_batch)

        # Log the loss value for monitoring
        logger.info(f"Validation Loss: {loss.item()}")

        # Get predicted class labels
        _, y_pred = torch.max(outputs, 1)

        # Collect predictions and true labels for metrics
        y_pred_list.extend(y_pred.cpu().numpy())
        y_true_list.extend(y_batch.cpu().numpy())

# Generate confusion matrix
cm = confusion_matrix(y_true_list, y_pred_list)

# Plot confusion matrix
labels = ['Class 0', 'Class 1']  # Adjust these according to your target classes
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Print classification report for more details
logger.info("Classification Report:\n" + classification_report(y_true_list, y_pred_list))

logger.info("Script completed successfully.")
