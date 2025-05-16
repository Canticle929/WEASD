import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Define constants
BASE_PATH = "/Volumes/xcy/TeamProject/WESAD/windowed_data/merged/"
ECG_FILE = os.path.join(BASE_PATH, "chest_ECG_windows.npy")
EDA_FILE = os.path.join(BASE_PATH, "chest_EDA_windows.npy")
RESP_FILE = os.path.join(BASE_PATH, "chest_Resp_windows.npy")
TEMP_FILE = os.path.join(BASE_PATH, "chest_Temp_windows.npy") # Potential temperature data
LABEL_FILE = os.path.join(BASE_PATH, "window_labels.npy")

# For now, sequence length is based on 5s window at 700Hz
SEQUENCE_LENGTH = 3500 
# Number of features will depend on included signals

def load_data(include_temp=False):
    """
    Loads windowed chest data (ECG, EDA, Resp, and optionally Temp) and labels.

    Args:
        include_temp (bool): Whether to include temperature data.

    Returns:
        tuple: A tuple containing:
            - data (np.ndarray): Combined sensor data of shape (n_samples, sequence_length, n_features).
            - labels (np.ndarray): Corresponding labels of shape (n_samples,).
    """
    try:
        ecg_windows = np.load(ECG_FILE)
        eda_windows = np.load(EDA_FILE)
        resp_windows = np.load(RESP_FILE)
        labels = np.load(LABEL_FILE)

        # Ensure all data has the same number of samples (windows)
        n_samples = ecg_windows.shape[0]
        if not (eda_windows.shape[0] == n_samples and resp_windows.shape[0] == n_samples and labels.shape[0] == n_samples):
            raise ValueError("Mismatch in the number of samples across data files.")

        # Ensure sequence length matches
        if not (ecg_windows.shape[1] == SEQUENCE_LENGTH and \
                eda_windows.shape[1] == SEQUENCE_LENGTH and \
                resp_windows.shape[1] == SEQUENCE_LENGTH):
            # Potentially try to reshape if it's (n_samples, features, sequence_length)
            # For now, strict check
            raise ValueError(f"Mismatch in sequence length. Expected {SEQUENCE_LENGTH}.")

        signals = [ecg_windows, eda_windows, resp_windows]
        if include_temp:
            temp_windows = np.load(TEMP_FILE)
            if temp_windows.shape[0] != n_samples:
                raise ValueError("Mismatch in the number of samples for temperature data.")
            if temp_windows.shape[1] != SEQUENCE_LENGTH:
                raise ValueError(f"Mismatch in sequence length for temperature data. Expected {SEQUENCE_LENGTH}.")
            signals.append(temp_windows)

        # Stack along a new axis to create (n_samples, sequence_length, n_features)
        # np.stack creates a new axis, then transpose to get features as the last dimension
        # Original shape for each signal: (n_samples, sequence_length)
        # We want to combine them so that for each sample, for each time step, we have a vector of sensor readings.
        # So, we stack them along a new dimension (axis=2)
        # Example: ecg (11012, 3500), eda (11012, 3500) -> stacked (11012, 3500, 2)
        
        data = np.stack(signals, axis=-1) # Results in (n_samples, sequence_length, n_features)
        
        print(f"Data loaded successfully. Shape: {data.shape}")
        print(f"Labels loaded successfully. Shape: {labels.shape}")
        
        # Ensure labels are integers (0 and 1, assuming they were 1 and 2 before)
        # WESAD labels are 1 (baseline) and 2 (stress). Let's map them to 0 and 1.
        labels = np.where(labels == 1, 0, labels) # Baseline (1) to 0
        labels = np.where(labels == 2, 1, labels) # Stress (2) to 1
        # Add checks for other labels if necessary, for now assuming only 1 and 2 are present for this binary task.
        
        # Filter out any labels not 0 or 1 if other states were present
        valid_indices = np.where((labels == 0) | (labels == 1))[0]
        data = data[valid_indices]
        labels = labels[valid_indices]
        
        print(f"Data shape after filtering for binary labels (0, 1): {data.shape}")
        print(f"Labels shape after filtering: {labels.shape}")


        return data, labels

    except FileNotFoundError as e:
        print(f"Error: One or more data files not found. {e}")
        return None, None
    except ValueError as e:
        print(f"Error loading data: {e}")
        return None, None

class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings, so that the two can be summed.
    Here, we use sine and cosine functions of different frequencies.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe) # pe is not a model parameter, but should be part of the state_dict

    def forward(self, x):
        """Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x.size(1) is the sequence length
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_encoder_layers, num_classes, dim_feedforward=2048, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.model_dim = model_dim
        self.input_fc = nn.Linear(input_dim, model_dim) # Project input features to model dimension
        self.pos_encoder = PositionalEncoding(model_dim, dropout, max_len=SEQUENCE_LENGTH)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Important: expects (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.output_fc = nn.Linear(model_dim, num_classes) # Output layer

    def forward(self, src):
        """Args:
            src: Tensor, shape [batch_size, seq_len, input_dim]
        """
        # Project input features to model dimension
        src = self.input_fc(src) # [batch_size, seq_len, model_dim]
        src = src * np.sqrt(self.model_dim) # Scale embedding (common practice)
        src = self.pos_encoder(src) # Add positional encoding
        
        # Pass through Transformer encoder
        # The nn.TransformerEncoderLayer and nn.TransformerEncoder expect no src_mask if self-attention.
        # If a mask is needed (e.g. for padding), it should be a boolean tensor where True indicates a value should be ignored.
        output = self.transformer_encoder(src) # [batch_size, seq_len, model_dim]
        
        # Use the output of the Transformer corresponding to the mean of the sequence
        # This is a common way to aggregate sequence information for classification
        output = output.mean(dim=1) # [batch_size, model_dim]
        
        # Pass through the final classification layer
        output = self.output_fc(output) # [batch_size, num_classes]
        return output

def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train() # Set model to training mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() # Zero the gradients
        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, labels) # Compute loss
        loss.backward() # Backward pass
        optimizer.step() # Update weights

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def evaluate_epoch(model, data_loader, criterion, device):
    model.eval() # Set model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad(): # No need to track gradients during evaluation
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

if __name__ == '__main__':
    # Example usage:
    # Decide whether to include temperature data
    USE_TEMP_DATA = False # Set to True to include temperature
    
    X, y = load_data(include_temp=USE_TEMP_DATA)

    if X is not None and y is not None:
        print(f"Original data shape: {X.shape}, Original labels shape: {y.shape}")
        print(f"Number of features: {X.shape[2]}")
        
        # 1. Data Splitting (Train, Validation, Test)
        # Split into training (70%) and temporary (30% for validation + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        
        # Split temporary (30%) into validation (15%) and test (15%) - (0.5 of 0.3 is 0.15)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
        
        print(f"Train set: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
        print(f"Validation set: X_val shape {X_val.shape}, y_val shape {y_val.shape}")
        print(f"Test set: X_test shape {X_test.shape}, y_test shape {y_test.shape}")

        # 2. Data Scaling (StandardScaler)
        # Scaler should be fit only on the training data's features.
        # Reshape data to 2D for scaler: (n_samples * sequence_length, n_features)
        num_features = X_train.shape[2]
        
        # Reshape X_train for fitting the scaler
        # (n_samples, sequence_length, n_features) -> (n_samples * sequence_length, n_features)
        X_train_reshaped = X_train.reshape(-1, num_features)
        
        scaler = StandardScaler()
        scaler.fit(X_train_reshaped) # Fit ONLY on training data
        
        # Transform training data
        X_train_scaled_reshaped = scaler.transform(X_train_reshaped)
        X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)
        
        # Transform validation data
        X_val_reshaped = X_val.reshape(-1, num_features)
        X_val_scaled_reshaped = scaler.transform(X_val_reshaped)
        X_val_scaled = X_val_scaled_reshaped.reshape(X_val.shape)
        
        # Transform test data
        X_test_reshaped = X_test.reshape(-1, num_features)
        X_test_scaled_reshaped = scaler.transform(X_test_reshaped)
        X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape)
        
        print("Data scaled successfully.")
        print(f"X_train_scaled shape: {X_train_scaled.shape}")
        print(f"X_val_scaled shape: {X_val_scaled.shape}")
        print(f"X_test_scaled shape: {X_test_scaled.shape}")

        # 3. Convert to PyTorch Tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long) # Using .long for CrossEntropyLoss, or .float for BCEWithLogitsLoss if output is single logit
        
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        print("Data converted to PyTorch tensors.")
        print(f"X_train_tensor shape: {X_train_tensor.shape}, y_train_tensor shape: {y_train_tensor.shape}")

        # 4. Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        BATCH_SIZE = 32 # Example batch size, can be tuned
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print("DataLoaders created.")
        print(f"Number of batches in train_loader: {len(train_loader)}")
        
        # 5. Define Transformer Model (placeholder for now)
        INPUT_DIM = X_train_tensor.shape[2] # Number of features
        MODEL_DIM = 64 # Embedding dimension / Transformer hidden size
        NUM_HEADS = 4  # Number of attention heads
        NUM_LAYERS = 2 # Number of Transformer encoder layers
        NUM_CLASSES = 2 # Baseline vs Stress (0 vs 1)

        model = TransformerClassifier(INPUT_DIM, MODEL_DIM, NUM_HEADS, NUM_LAYERS, NUM_CLASSES)
        criterion = nn.CrossEntropyLoss() # If NUM_CLASSES > 1 and output is raw scores
        # criterion = nn.BCEWithLogitsLoss() # If NUM_CLASSES = 1 (or 2 with last layer sigmoid and BCELoss)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print("Model, criterion, and optimizer defined.")

        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model.to(device)

        # 6. Training Loop
        NUM_EPOCHS = 10 # Example number of epochs, can be tuned

        print("\nStarting training...")
        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)

            print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        print("\nTraining finished.")

        # 7. Evaluation on Test Set
        print("\nEvaluating on test set...")
        test_loss, test_acc = evaluate_epoch(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        # 8. Save the model (optional)
        # Ensure WESAD directory exists
        # output_model_dir = "/Volumes/xcy/TeamProject/WESAD/Transformer/"
        # os.makedirs(output_model_dir, exist_ok=True)
        # output_model_path = os.path.join(output_model_dir, "transformer_chest_model.pth")
        # torch.save(model.state_dict(), output_model_path)
        # print(f"Model saved to {output_model_path}")
        
        pass 