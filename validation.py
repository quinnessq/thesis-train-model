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
VALIDATION_DATA_PATH = r'C:\Users\alcui\Desktop\MSCE\Modules\Afstuderen\trainingdata\validation.csv'  # Path to CSV file
MODEL_PATH = 'lstm_model.pth'  # Path to save the model
TARGET_COLUMN = 'malicious'  # Target variable for classification
BATCH_SIZE = 32  # Batch size for training
SEQUENCE_LENGTH = 10  # Number of time steps per input sequence

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Load and preprocess data
logger.info("Loading data...")
data = pd.read_csv(
    VALIDATION_DATA_PATH,
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
    data = data.sample(n=100, random_state=42)  # Limit data to 100 samples
    logger.info("TEST_MODE is ON. Using a limited data subset.")

# Sort by time to preserve chronological order
data = data.sort_values(by='time')

# Some data transformations from str to labels that are usable
le = LabelEncoder()
data['method_encoded'] = le.transform(data['request_method'])
data['protocol_encoded'] = le.transform(data['protocol'])
data['referrer_encoded'] = le.transform(data['referrer'])
data['request_content_type_encoded'] = le.transform(data['request_content_type'])
data['remote_ip_int'] =  data['remote_ip'].apply(lambda ip: int(ip.replace('.', '')))
data['user_agent_encoded'] = le.transform(data['user_agent'])  # Add user_agent encoding
data['request_uri_encoded'] = le.transform(data['request_uri'])  # Add user_agent encoding

# Drop original columns if no longer needed
data = data.drop(columns=['date', 'request_method', 'protocol', 'referrer', 'user_agent', 'request_uri', 'remote_ip', 'request_content_type'])

# Define features and target
feature_columns = [col for col in data.columns if col != TARGET_COLUMN]
x = data[feature_columns].values
y = data[TARGET_COLUMN].values

# Convert to sequences using numpy.array() to avoid slow list-to-tensor conversion
def create_sequences(sequence_x, sequence_y, sequence_length):
    sequences = np.array([sequence_x[i:i + sequence_length] for i in range(len(sequence_x) - sequence_length + 1)])
    labels = np.array([sequence_y[i + sequence_length - 1] for i in range(len(sequence_y) - sequence_length + 1)])
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# Prepare sequential data
X_seq, y_seq = create_sequences(x, y, SEQUENCE_LENGTH)

# Create DataLoader for validation data
validation_dataset = TensorDataset(X_seq, y_seq)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
model = LSTMModel(input_size, hidden_size=64, output_size=2, dropout_rate=0.2).to(device)

# Load the pre-trained model if available
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    logger.info("Loaded existing model from disk.")
else:
    logger.error("No model found at the specified path.")
    exit()

# Model evaluation
logger.info("Evaluating the model...")
model.eval()
y_pred_list = []
y_true_list = []

with torch.no_grad():
    for X_batch, y_batch in validation_loader:
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
