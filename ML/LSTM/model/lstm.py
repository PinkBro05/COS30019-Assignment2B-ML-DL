import numpy as np
import torch
import torch.nn as nn
from .lstm_cell import LSTMCell

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create a complete LSTM layer using our custom LSTM Cell
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, input_size) if batch_first
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
        else:
            seq_len, batch_size, _ = x.size()
            x = x.transpose(0, 1)  # Convert to batch_first

        # Initialize hidden state if not provided
        if hidden is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)
            hidden = (h, c)

        # Process sequence
        outputs = []
        h, c = hidden

        for t in range(seq_len):
            x_t = x[:, t, :]  # Get input at time t
            h, c = self.cell(x_t, (h, c))
            outputs.append(h)

        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_size)

        return outputs, (h, c)