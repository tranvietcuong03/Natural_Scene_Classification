import torch
import torch.nn as nn

class MyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = self._make_block(3, 16)
        self.conv2 = self._make_block(16, 32)
        self.conv3 = self._make_block(32, 64)
        self.conv4 = self._make_block(64, 128)
        self.conv5 = self._make_block(128, 128)

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=128*6*6, out_features=2048),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), -1) # flattern
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    def _make_block(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3,stride=1, padding='same'),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3,stride=1, padding='same'),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
