import torch
import torch.nn as nn
import torch.nn.functional as F

class KernelAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class KAN(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = KernelAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# Example usage
input_dim = 256
kan_layer = KAN(dim=input_dim)

# Replace MLP with KAN
class ModelWithKAN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.kan = KAN(dim=input_dim)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.kan(x)
        return self.fc(x)

# Create and use the model
model = ModelWithKAN(input_dim=256, output_dim=10)
input_tensor = torch.randn(32, 100, 256)  # (batch_size, sequence_length, input_dim)
output = model(input_tensor)
print(output.shape)  # Should print torch.Size([32, 100, 10])