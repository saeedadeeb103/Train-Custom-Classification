import torch
import torch.nn as nn

class CTCEncoder(nn.Module):
    def __init__(self, num_classes, cnn_output_dim=256, rnn_hidden_dim=256, rnn_layers=3):
        """
        CTC Encoder with a CNN feature extractor and LSTM for sequence modeling.

        Args:
            num_classes (int): Number of output classes for the model.
            cnn_output_dim (int): Number of output channels from the CNN.
            rnn_hidden_dim (int): Hidden size of the LSTM.
            rnn_layers (int): Number of layers in the LSTM.
        """
        super(CTCEncoder, self).__init__()

        # CNN Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Down-sample by 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Down-sample by another 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))  # Ensure output height is 1
        )

        # Bidirectional LSTM
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_layers = rnn_layers
        self.cnn_output_dim = cnn_output_dim

        self.rnn = nn.LSTM(
            input_size=cnn_output_dim,  # Output channels from CNN
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True
        )

        # Fully connected layer
        self.fc = nn.Linear(rnn_hidden_dim * 2, num_classes)

    def compute_input_lengths(self, input_lengths):
        """
        Adjusts input lengths based on the CNN's down-sampling operations.

        Args:
            input_lengths (torch.Tensor): Original input lengths.

        Returns:
            torch.Tensor: Adjusted input lengths.
        """
        # Account for down-sampling by MaxPool layers (factor of 2 for each MaxPool)
        input_lengths = input_lengths // 2  # First MaxPool
        input_lengths = input_lengths // 2  # Second MaxPool
        input_lengths = input_lengths // 2  # Third pooling layer or additional down-sampling
        return input_lengths

    def forward(self, x, input_lengths):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape [B, 1, H, W].
            input_lengths (torch.Tensor): Lengths of the sequences in the batch.

        Returns:
            torch.Tensor: Logits of shape [B, T, num_classes].
            torch.Tensor: Adjusted input lengths.
        """
        # Feature extraction
        x = self.feature_extractor(x)  # [Batch_Size, Channels, Height, Width]
        print(f"Shape after CNN: {x.shape}")  # Debug the shape

        # Reshape for LSTM
        x = x.squeeze(2).permute(0, 2, 1)  # [Batch_Size, Sequence_Length, Features]
        assert x.size(-1) == 256, f"Expected last dimension to be 256, but got {x.size(-1)}"

        # Adjust input lengths
        input_lengths = self.compute_input_lengths(input_lengths)
        assert input_lengths.size(0) == x.size(0), f"input_lengths size ({input_lengths.size(0)}) must match batch size ({x.size(0)})"

        # Pass through LSTM
        x, _ = self.rnn(x)  # [Batch_Size, Sequence_Length, 2 * Hidden_Dim]

        # Fully connected output
        x = self.fc(x)  # [Batch_Size, Sequence_Length, Num_Classes]
        return x, input_lengths
