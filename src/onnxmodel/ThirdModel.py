import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from .BaseModel import BaseModel
import matplotlib.pyplot as plt
import numpy as np
import json
import datetime
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image


class ThirdModel(BaseModel):
    def __init__(
        self,
        batch_size=64,  # Smaller batch size for better generalization
        learning_rate=0.0003,  # Lower initial learning rate
        num_epochs=30,  # More epochs for better convergence
        dataset_name="imagenet",  # Only for ImageNet
    ):
        if dataset_name.lower() != "imagenet":
            raise ValueError("ThirdModel is specifically designed for ImageNet")

        super().__init__(
            batch_size, learning_rate, num_epochs, dataset_name=dataset_name
        )
        self.build_model()

    def get_transforms(self, is_train=True):
        """
        Get the data transformations for ImageNet.
        Enhanced data augmentation for training.
        """
        if is_train:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(64),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    ),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize(72),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def build_model(self):
        """
        Build an efficient CNN model optimized for ImageNet.
        """

        class ConvBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
                super().__init__()
                padding = kernel_size // 2
                self.conv = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x):
                return self.conv(x)

        class SEBlock(nn.Module):
            def __init__(self, channels, reduction=16):
                super().__init__()
                self.squeeze = nn.AdaptiveAvgPool2d(1)
                self.excitation = nn.Sequential(
                    nn.Linear(channels, channels // reduction, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(channels // reduction, channels, bias=False),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                b, c, _, _ = x.size()
                y = self.squeeze(x).view(b, c)
                y = self.excitation(y).view(b, c, 1, 1)
                return x * y.expand_as(x)

        class EfficientBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()
                self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)
                self.conv2 = ConvBlock(out_channels, out_channels, kernel_size=3)
                self.se = SEBlock(out_channels)
                self.shortcut = nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                        nn.BatchNorm2d(out_channels),
                    )

            def forward(self, x):
                out = self.conv1(x)
                out = self.conv2(out)
                out = self.se(out)
                out += self.shortcut(x)
                return F.relu(out)

        class EfficientNet(nn.Module):
            def __init__(self, num_classes=200):
                super().__init__()
                self.features = nn.Sequential(
                    ConvBlock(3, 64),
                    EfficientBlock(64, 128, stride=2),
                    EfficientBlock(128, 256, stride=2),
                    EfficientBlock(256, 512, stride=2),
                )

                self.avgpool = nn.AdaptiveAvgPool2d(1)
                self.classifier = nn.Sequential(
                    nn.Dropout(0.3), nn.Linear(512, num_classes)
                )

                # Initialize weights
                self.apply(self._init_weights)

            def _init_weights(self, m):
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x

        self.net = EfficientNet().to(self.device)

        # Initialize optimizer with weight decay
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )

    def train(self) -> tuple:
        """
        Train the model with improved monitoring and logging.
        """
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        best_val_acc = 0.0
        patience = 5
        patience_counter = 0

        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3, verbose=True
        )

        for epoch in range(self.num_epochs):
            # Training phase
            self.net.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for i, (inputs, labels) in enumerate(self.trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if (i + 1) % 50 == 0:
                    print(
                        f"Epoch [{epoch+1}/{self.num_epochs}], "
                        f"Step [{i+1}/{len(self.trainloader)}], "
                        f"Loss: {running_loss/50:.4f}, "
                        f"Acc: {100.*correct/total:.2f}%"
                    )
                    running_loss = 0.0

            # Validation phase
            self.net.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in self.valloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            # Calculate metrics
            epoch_val_loss = val_loss / len(self.valloader)
            epoch_val_acc = 100.0 * val_correct / val_total
            epoch_train_acc = 100.0 * correct / total
            epoch_train_loss = running_loss / len(self.trainloader)

            # Save metrics
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)

            # Update learning rate
            scheduler.step(epoch_val_acc)

            # Save best model
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                patience_counter = 0
                save_dir = os.path.join(os.getcwd(), "..", "..", "saved_models")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(
                    self.net.state_dict(),
                    os.path.join(save_dir, "best_third_model.pth"),
                )
                print(
                    f"New best model saved with validation accuracy: {best_val_acc:.2f}%"
                )
            else:
                patience_counter += 1

            print(
                f"Epoch [{epoch+1}/{self.num_epochs}]: "
                f"Train Loss: {epoch_train_loss:.4f}, "
                f"Train Acc: {epoch_train_acc:.2f}%, "
                f"Val Loss: {epoch_val_loss:.4f}, "
                f"Val Acc: {epoch_val_acc:.2f}%"
            )

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        print("Training completed!")
        return train_losses, train_accuracies, val_losses, val_accuracies

    def predict_single_image(self, image_path: str, label_mapping=None):
        """
        Predict class for a single image.
        """
        self.net.eval()
        transform = self.get_transforms(is_train=False)

        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.net(image)
            _, predicted = outputs.max(1)

        predicted_class = (
            label_mapping[predicted.item()]
            if label_mapping
            else f"Class {predicted.item()}"
        )
        print(f"Predicted Class: {predicted_class}")
