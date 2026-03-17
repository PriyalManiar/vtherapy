import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

class ResNeXtLSTMBinary(nn.Module):
    def __init__(self, base_model, hidden_dim, lstm_layers, bidirectional, dropout_prob):
        super(ResNeXtLSTMBinary, self).__init__()
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        self.avgpool = base_model.avgpool
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 2)

    def forward(self, x):
        batch_size, seq_length, C, H, W = x.size()
        x = x.view(batch_size * seq_length, C, H, W)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = x.view(batch_size, seq_length, -1)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(lstm_out)
        return out
