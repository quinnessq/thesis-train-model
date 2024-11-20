import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report

# Configuration
VALIDATION_DATA_PATH = r'C:\Users\alcui\Desktop\MSCE\Modules\Afstuderen\trainingdata\validation.csv'  # Path to CSV file
PROCESSED_DATA_VALIDATION_PATH = r'C:\Users\alcui\Desktop\MSCE\Modules\Afstuderen\trainingdata\processed_data_validation.pkl'  # Path for processed data

TARGET_COLUMN = 'malicious'  # Target variable for classification
MODEL_PATH = 'lstm_model.pth'

BATCH_SIZE = 8192
HIDDEN_SIZE = 64
SEQUENCE_LENGTH = 8192

DROPOUT_RATE = 0.2
OUTPUT_SIZE = 2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Check if processed data exists and load it, otherwise process raw data
logger.info("Loading processed data from file...")
data = pd.read_pickle(PROCESSED_DATA_VALIDATION_PATH)

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
input_size = len(feature_columns)
logger.info(f"Feature column size: {input_size}")
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
    hidden = None  # Initialize the hidden state for evaluation
    for X_batch, y_batch in data_generator():
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs, hidden = model(X_batch, hidden)
        hidden = tuple([h.detach() for h in hidden])
        loss = criterion(outputs, y_batch)
        _, y_pred = torch.max(outputs, 1)
        y_pred_list.extend(y_pred.cpu().numpy())
        y_true_list.extend(y_batch.cpu().numpy())


# Generate confusion matrix
cm = confusion_matrix(y_true_list, y_pred_list)

# Plot confusion matrix
labels = ['Benign', 'Malicious']  # Adjust these according to your target classes
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Print classification report for more details
logger.info("Classification Report:\n" + classification_report(y_true_list, y_pred_list))

logger.info("Script completed successfully.")
