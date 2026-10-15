"""
The symbol classifier: a small convolutional neural network for 28x28 crops.
"""
import torch.nn as nn

# the symbols the app needs, in the order the network outputs them
CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "x", "y", "+", "-", "="]


class SymbolNet(nn.Module):
    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # 16 filters looking for small patterns (edges, curves)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 28x28 -> 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 32 filters combining those into bigger shapes
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.3),                              # randomly ignores some neurons while training, to reduce overfitting
            nn.Linear(64, num_classes),                   # one score per symbol
        )

    def forward(self, x):
        return self.classifier(self.features(x))
