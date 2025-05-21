import numpy as np
import torch
import torch.nn as nn

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the custom LSTM Cell as provided
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # Define weights for gates
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        # Concatenate input and previous hidden state
        combined = torch.cat((x, h_prev), dim=1)

        # Calculate gate activations
        f = torch.sigmoid(self.forget_gate(combined))
        i = torch.sigmoid(self.input_gate(combined))
        o = torch.sigmoid(self.output_gate(combined))
        c_tilde = torch.tanh(self.cell_gate(combined))

        # Update cell state and hidden state
        c_next = f * c_prev + i * c_tilde
        h_next = o * torch.tanh(c_next)

        return h_next, c_next