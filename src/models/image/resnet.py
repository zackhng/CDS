import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4  # output channels multiplier

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels,
                               out_channels * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        # shortcut projection if shape changes
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * self.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class ResNet50(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.in_channels = 64

        # Initial stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1)

        # ResNet stages
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        # classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion,
                            num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []

        # first block may downsample
        layers.append(
            Bottleneck(self.in_channels,
                       out_channels,
                       stride)
        )

        self.in_channels = out_channels * Bottleneck.expansion

        for _ in range(1, blocks):
            layers.append(
                Bottleneck(self.in_channels,
                           out_channels)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        x: [B, T, C, H, W]  (batch of videos)
        returns: [B, num_classes]  (one prediction per video)
        """
        if x.dim() != 5:
            raise ValueError(f"Expected input 5D [B,T,C,H,W], got {x.shape}")

        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # merge batch and frames

        # pass through ResNet CNN layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # shape: [B*T, feature_dim]

        # restore batch + frames
        x = x.view(B, T, -1)  # [B, T, feature_dim]

        # temporal pooling: mean over frames
        x = x.mean(dim=1)  # [B, feature_dim]

        # classifier
        x = self.fc(x)  # [B, num_classes]

        return x