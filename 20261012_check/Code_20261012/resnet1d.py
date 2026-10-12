import torch
import torch.nn as nn

class SqueezeExcitation1D(nn.Module):
    """Dynamically weights ECG channels based on global context."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.fc(x).unsqueeze(-1)
        return x * weight


class MultiScaleConv1D(nn.Module):
    """Extracts features across perfectly balanced parallel channel blocks."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Divide channels perfectly into 4 distinct groups to prevent split errors
        branch_features = out_channels // 4
        
        self.branch1 = nn.Conv1d(in_channels, branch_features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.branch2 = nn.Conv1d(in_channels, branch_features, kernel_size=5, stride=stride, padding=2, bias=False)
        self.branch3 = nn.Conv1d(in_channels, branch_features, kernel_size=7, stride=stride, padding=3, bias=False)
        self.branch4 = nn.Conv1d(in_channels, branch_features, kernel_size=11, stride=stride, padding=5, bias=False)
        
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class ImprovedBasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = MultiScaleConv1D(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SqueezeExcitation1D(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            # Use standard stride routing to keep identity tracking synchronized
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        return self.relu(out)


class ResNet1D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Receptive stem processing layer
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Power of 2 channels keep spatial sequence math completely matching
        self.layer1 = ImprovedBasicBlock1D(64, 64)
        self.layer2 = ImprovedBasicBlock1D(64, 128, stride=2)
        self.layer3 = ImprovedBasicBlock1D(128, 256, stride=2)
        self.layer4 = ImprovedBasicBlock1D(256, 512, stride=2)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        return self.fc(x)
