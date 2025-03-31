# First approach for the image recognition NN Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from .BaseModel import BaseModel

"""
Structure:
- 2 convolutional layers with max pooling
- 3 fully connected layers
- Uses ReLU activation for hidden layers
- Output layer with softmax activation for probability distribution over 10 classes
"""


class FirstModel(BaseModel):
    def __init__(
        self, batch_size=4, learning_rate=0.001, num_epochs=20, dataset_name="cifar10"
    ):
        super().__init__(batch_size, learning_rate, num_epochs, dataset_name="cifar10")
        self.build_model()

    def get_transforms(self):
        return super().get_transforms()

    def build_model(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(
                    3, 6, 5
                )  # Output: (6, 28, 28) -> Input: (3, 32, 32), kernel: 5x5, no padding
                self.pool = nn.MaxPool2d(
                    2, 2
                )  # Output: (6, 14, 14) after first pooling
                self.conv2 = nn.Conv2d(
                    6, 16, 5
                )  # Output: (16, 10, 10) -> Input: (6, 14, 14), kernel: 5x5, no padding
                # After second pooling: (16, 5, 5)
                self.fc1 = nn.Linear(
                    16 * 5 * 5, 120
                )  # Input: 16*5*5 = 400, Output: 120
                self.fc2 = nn.Linear(120, 84)  # Input: 120, Output: 84
                self.fc3 = nn.Linear(
                    84, 10
                )  # Input: 84, Output: 10 (number of classes)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))  # Output: (6, 14, 14)
                x = self.pool(F.relu(self.conv2(x)))  # Output: (16, 5, 5)
                x = torch.flatten(x, 1)  # Output: 16 * 5 * 5 = 400
                x = F.relu(self.fc1(x))  # Output: 120
                x = F.relu(self.fc2(x))  # Output: 84
                x = self.fc3(x)  # Output: 10
                return x

        self.net = Net().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate)
