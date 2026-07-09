import torch
import torch.nn as nn


class MultiLayerNN(nn.Module):
    """
    Multi-layer Neural Network with hidden layers
    
    Architecture:
        Input (4096) -> Hidden1 (500) -> Hidden2 (250) -> Output (1)
    
    This allows the network to learn compressed intermediate representations,
    potentially discovering features beyond what explicit semantic axes capture.
    """
    
    def __init__(self, input_dim=4096, hidden1_dim=500, hidden2_dim=250, dropout=0.3):
        super(MultiLayerNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        
        # Layer 1: Input -> Hidden1
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.bn1 = nn.BatchNorm1d(hidden1_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer 2: Hidden1 -> Hidden2
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.bn2 = nn.BatchNorm1d(hidden2_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # Output layer: Hidden2 -> Output
        self.fc3 = nn.Linear(hidden2_dim, 1)
        
        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, return_logits=False):
        """Forward pass through the network

        Args:
            x: Input tensor
            return_logits: If True, return raw logits (before sigmoid).
                          If False (default), return probabilities.
        """
        # Hidden layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Hidden layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # Output layer
        x = self.fc3(x)

        if return_logits:
            return x
        return self.sigmoid(x)
    
    def get_hidden_representation(self, x, layer=1):
        """
        Extract hidden layer representations

        Args:
            x: Input tensor
            layer: Which hidden layer to extract (1 or 2)

        Returns:
            Hidden layer activations
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            # Layer 1
            h1 = self.fc1(x)
            h1 = self.bn1(h1)
            h1 = self.relu(h1)

            if layer == 1:
                if was_training:
                    self.train()
                return h1

            # Layer 2
            h2 = self.fc2(h1)
            h2 = self.bn2(h2)
            h2 = self.relu(h2)

            if was_training:
                self.train()
            return h2

