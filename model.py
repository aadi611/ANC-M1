import torch
import torch.nn as nn
from typing import Tuple

class DoubleConv(nn.Module):
    """Double convolution block with BatchNorm, LeakyReLU, and Dropout."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class UNetANC(nn.Module):
    """UNet-based Audio Noise Cancellation model for 1D audio signals."""
    
    def __init__(self, in_channels: int = 1, base_channels: int = 64, dropout: float = 0.2):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, base_channels, dropout)
        self.enc2 = DoubleConv(base_channels, base_channels * 2, dropout)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4, dropout)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8, dropout)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 8, dropout)
        
        # Decoder
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 4, dropout)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 2, dropout)
        self.dec2 = DoubleConv(base_channels * 3, base_channels, dropout)
        
        # Final output layer
        self.dec1 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_channels // 2, in_channels, kernel_size=1)
        )
        
        # Pooling and upsampling
        self.pool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through UNet architecture.
        
        Args:
            x: Input tensor of shape (batch, 1, length)
            
        Returns:
            Denoised audio tensor of shape (batch, 1, length)
        """
        # Encoder with skip connections
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.dec4(torch.cat([self.upsample(bottleneck), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.upsample(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upsample(dec3), enc2], dim=1))
        
        # Final output
        out = self.upsample(dec2)
        out = self.dec1(out)
        
        return torch.tanh(out)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size(self) -> Tuple[int, str]:
        """Get model size in bytes and human-readable format."""
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        total_size = param_size + buffer_size
        
        # Convert to MB
        size_mb = total_size / (1024 ** 2)
        return total_size, f"{size_mb:.2f} MB"

# Example usage
if __name__ == "__main__":
    model = UNetANC()
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Model size: {model.get_model_size()[1]}")
    
    # Test forward pass
    x = torch.randn(2, 1, 32000)  # Batch of 2, 1 channel, 32000 samples
    with torch.no_
