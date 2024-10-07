import torch.nn as nn
import torch
import torch.nn.functional as F
from math import ceil

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size, reduction_ratio=4):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        
        # Reduce hidden_dim by adjusting the expand_ratio
        hidden_dim = max(1, in_channels * expand_ratio // 2)
        
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm2d(hidden_dim)
        else:
            self.expand_conv = None
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(hidden_dim if expand_ratio != 1 else in_channels, hidden_dim, 
                                        kernel_size=kernel_size, stride=stride, 
                                        padding=kernel_size // 2, groups=hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        # Squeeze and Excitation layers
        self.se_avgpool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Conv2d(hidden_dim, max(1, hidden_dim // reduction_ratio), kernel_size=1)
        self.se_fc2 = nn.Conv2d(max(1, hidden_dim // reduction_ratio), hidden_dim, kernel_size=1)
        
        # Pointwise projection layer
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.use_residual = (in_channels == out_channels and stride == 1)
    
    def forward(self, x):
        identity = x
        
        # Expansion layer
        if self.expand_conv:
            out = F.relu6(self.bn0(self.expand_conv(x)))
        else:
            out = x
        
        # Depthwise convolution
        out = F.relu6(self.bn1(self.depthwise_conv(out)))
        
        # Squeeze and Excitation
        se = self.se_avgpool(out)
        se = F.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        out = out * se
        
        # Projection layer
        out = self.bn2(self.project_conv(out))
        
        # Residual connection
        if self.use_residual:
            out = out + identity
        
        return out


class ClassificationModel(nn.Module):
    def __init__(self, width_coefficient=0.13, depth_coefficient=0.5, dropout_rate=0.2, num_classes=100):
        super(ClassificationModel, self).__init__()
        
        base_channels = 8  # Reduced base channels to 8
        base_layers = [
            (1, 16, 1, 1, 3),   # (expand_ratio, out_channels, num_blocks, stride, kernel_size)
            (3, 24, 1, 2, 3),   
            (3, 40, 1, 2, 3),   
            (3, 80, 2, 2, 3),   
            (3, 112, 1, 1, 3),  
            (3, 192, 1, 2, 3),  
            (3, 320, 1, 1, 3)   
        ]
        
        # Reduce out_channels with width_coefficient
        out_channels = max(1, ceil(base_channels * width_coefficient))
        self.stem_conv = nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(out_channels)
        
        self.blocks = nn.ModuleList([])
        in_channels = out_channels
        for expand_ratio, out_channels, num_blocks, stride, kernel_size in base_layers:
            out_channels = max(1, ceil(out_channels * width_coefficient))
            num_blocks = max(1, ceil(num_blocks * depth_coefficient))
            for i in range(num_blocks):
                block_stride = stride if i == 0 else 1
                self.blocks.append(MBConvBlock(in_channels, out_channels, expand_ratio, block_stride, kernel_size))
                in_channels = out_channels

        # Reduce final_channels with width_coefficient
        final_channels = max(1, ceil(640 * width_coefficient))
        self.head_conv = nn.Conv2d(in_channels, final_channels, kernel_size=1, bias=False)
        self.head_bn = nn.BatchNorm2d(final_channels)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(final_channels, num_classes)

    def forward(self, x):
        x = F.relu6(self.stem_bn(self.stem_conv(x)))
        
        for block in self.blocks:
            x = block(x)
        
        x = F.relu6(self.head_bn(self.head_conv(x)))
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

        
       
