import torch 
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank, alpha):
        super(self).__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(input_dim, rank) * std_dev)  # Low-rank matrix A
        self.B = nn.Parameter(torch.zeros(rank, output_dim))  # Low-rank matrix B
        self.alpha = alpha  # Scaling factor
    def forward(self, x):
        # Apply low-rank adaptation: x + alpha * (x @ A @ B)
        return self.alpha * (x @ self.A @ self.B)
    

class LinearWithLoRA(nn.Module):
    def __init__(self, linear_layer, rank, alpha):
        super().__init__()
        self.linear = linear_layer  # Original linear layer
        self.lora = LoRALayer(linear_layer.in_features, linear_layer.out_features, rank, alpha)  # LoRA layer

    def forward(self, x):
        # Combine original linear layer output with LoRA adaptation
        return self.linear(x) + self.lora(x)