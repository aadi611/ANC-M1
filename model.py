
import torch 
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetANC(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(1, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 512)
        self.dec4 = DoubleConv(512 + 512, 256)
        self.dec3 = DoubleConv(256 + 256, 128)
        self.dec2 = DoubleConv(128 + 64, 64)
        self.dec1 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=1)
        )
        self.pool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        
        x = self.bottleneck(self.pool(x4))
        
        x = self.dec4(torch.cat([self.upsample(x), x4], dim=1))
        x = self.dec3(torch.cat([self.upsample(x), x3], dim=1))
        x = self.dec2(torch.cat([self.upsample(x), x2], dim=1))
        x = self.upsample(x)
        x = self.dec1(x)

        return torch.tanh(x)

