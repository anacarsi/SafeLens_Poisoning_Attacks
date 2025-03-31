# FoPra IT Sicherheit WS 2024/25
# Second approach for the image recognition NN

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
from torch.optim.lr_scheduler import StepLR
from PIL import Image  # to load and convert the image into RGB format
from torchvision.transforms import v2

"""
Structure:
CIFAR-10: Images are 32 x 32 x 3.
Tiny ImageNet: Images are 64 x 64 x 3.
For CIFAR-10:
    Output after convolutional layers: 128x4x4=2048.
    fc1 input size: 2048.
For Tiny ImageNet:
    Output after convolutional layers: 128x8x8=8192.
    fc1 input size: 8192.

Droupout layers and Adam optimization.
"""


class SecondModel(BaseModel):
    def __init__(
        self,
        batch_size=128,
        learning_rate=0.001,
        num_epochs=50,
        dataset_name="imagenet",
    ):
        super().__init__(
            batch_size, learning_rate, num_epochs, dataset_name=dataset_name
        )
        self.dataset_name = dataset_name
        self.build_model()
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.gradient_norms = []
        self.weight_updates = []

    def get_transforms(self, is_train=True) -> transforms.Compose:
        """
        Get the data transformations for the dataset.
        """
        return super().get_transforms(is_train=is_train)

    def count_params(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def build_model(self):
        """
        Build the neural network model.
        """

        class Net(nn.Module):
            def __init__(self, input_size, dataset_name="imagenet"):
                super(Net, self).__init__()
                self.input_size = input_size
                self.dataset_name = dataset_name
                # Convolutional layers
                self.conv1 = nn.Conv2d(
                    3, 32, kernel_size=3, padding=1
                )  # 3x32x32 -> 32x32x32
                self.conv2 = nn.Conv2d(
                    32, 64, kernel_size=3, padding=1
                )  # 32x16x16 -> 64x16x16
                self.pool = nn.MaxPool2d(2, 2)  # Halves spatial dimensions
                self.conv3 = nn.Conv2d(
                    64, 128, kernel_size=3, padding=1
                )  # 64x8x8 -> 128x8x8

                # Fully connected layers
                self.fc1 = nn.LazyLinear(out_features=256)  # Initialized dynamically
                self.fc2 = nn.Linear(256, 128)
                if self.dataset_name.lower() == "cifar10":
                    self.fc3 = nn.Linear(128, 10)
                elif self.dataset_name.lower() == "imagenet":
                    self.fc3 = nn.Linear(128, 200)
                else:
                    raise ValueError(
                        "Unsupported dataset type. Choose 'cifar10' or 'imagenet'"
                    )
                self.dropout = nn.Dropout(0.5)
                self.dropout2 = nn.Dropout(0.2)  # second dropout for conv layers

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Convolutional layers with ReLU and pooling
                x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> Pool
                x = self.dropout2(x)
                x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> Pool
                x = self.dropout2(x)
                x = self.pool(F.relu(self.conv3(x)))  # Conv3 -> Pool
                x = self.dropout2(x)
                # Flatten the tensor for fully connected layers
                x = torch.flatten(x, 1)

                # Fully connected layers pass
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        input_size = (
            32
            if self.dataset_name.lower() == "cifar10"
            else 64 if self.dataset_name.lower() == "imagenet" else None
        )
        self.net = Net(input_size, dataset_name=self.dataset_name).to(self.device)

        # Perform a dummy forward pass to initialize `fc1` dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_size, input_size).to(self.device)
            _ = self.net(dummy_input)
            # Check no errors
            print("Dummy model built successfully.")

        # self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def imshow(self, img: torch.Tensor):
        """
        Helper function to unnormalize and display an image.

        Parameters:
        - img: Image tensor
        """
        img = img / 2 + 0.5  # Unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis("off")
        plt.show()

    def top_k_accuracy(self, k=5):
        """
        Calculate the top-k accuracy of the model.
        """
        self.net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:  # data[0] is the image, data[1] is the label
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(images)
                _, predicted = torch.topk(outputs, k, 1)
                total += labels.size(0)
                for i in range(k):
                    correct += (predicted[:, i] == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Top-{k} Accuracy: {accuracy:.2f}%")
        return accuracy

    def load_imagenet_labels(self, file_path):
        """
        Load the ImageNet class-to-synset mapping from a JSON or text file

        Parameters:
        - file_path: Path to the file containing the mapping
        """
        with open(file_path, "r") as f:
            return json.load(f)

    def train(self, label_file_path=None) -> tuple:
        """
        Train the neural network.
        """
        train_losses = []
        train_accuracies = []

        total_batches = len(self.trainloader)
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Total batches per epoch: {total_batches}")

        self.net.train()
        scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Calculate accuracy
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Print progress every 100 batches
                if (i + 1) % 100 == 0:
                    current_acc = 100 * correct / total
                    avg_loss = running_loss / (i + 1)
                    print(
                        f"Batch {i + 1}/{total_batches} | "
                        f"Loss: {avg_loss:.3f} | "
                        f"Accuracy: {current_acc:.2f}% | "
                        f"LR: {scheduler.get_last_lr()[0]:.6f}"
                    )
                    learning_rate = scheduler.get_last_lr()[0]

            # Epoch summary
            epoch_loss = running_loss / len(self.trainloader)
            epoch_acc = 100 * correct / total
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)
            self.learning_rates.append(learning_rate)
            # Update scheduler
            scheduler.step()

            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Loss: {epoch_loss:.3f} | Accuracy: {epoch_acc:.2f}%")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        print("\nTraining completed!")

        file_name = (
            f"secondmodel_{self.dataset_name}_noval_{self.num_epochs}_metrics.csv"
        )
        self.write_metrics_to_file(file_path=os.path.join(os.getcwd(), file_name))
        print(f"Metrics saved to {file_name}")

        # Save final model
        save_dir = os.path.join(os.getcwd(), "..", "..", "saved_models")
        os.makedirs(save_dir, exist_ok=True)

        filename = f"final_second_model_{self.dataset_name.lower()}.pth"
        filepath = os.path.join(save_dir, filename)
        torch.save(self.net.state_dict(), filepath)
        print(f"Final model saved to {filepath}")

        return train_losses, train_accuracies

    def train_validation(self) -> tuple:
        """
        Train with improved learning rate scheduling and monitoring.
        """
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        # Add progress tracking
        total_batches = len(self.trainloader)
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Total batches per epoch: {total_batches}")

        self.net.train()
        scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=self.num_epochs,
            verbose=False,
        )
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     self.optimizer,
        #     max_lr=self.learning_rate,
        #     epochs=self.num_epochs,
        #     steps_per_epoch=len(self.trainloader),
        #     pct_start=0.3,
        #     div_factor=25,
        #     final_div_factor=1000,
        # )

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            # Training phase with progress tracking
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("Training phase:")

            for i, data in enumerate(self.trainloader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()
                scheduler.step()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()

                # Print progress every 100 batches
                if (i + 1) % 100 == 0:
                    current_acc = 100 * correct / total
                    avg_loss = running_loss / (i + 1)
                    print(
                        f"Batch {i + 1}/{total_batches} | "
                        f"Loss: {avg_loss:.3f} | "
                        f"Accuracy: {current_acc:.2f}% | "
                        f"LR: {scheduler.get_last_lr()[0]:.6f}"
                    )
                    learning_rate = scheduler.get_last_lr()[0]

            # Epoch training stats
            epoch_loss = running_loss / len(self.trainloader)
            epoch_acc = 100 * correct / total
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)
            self.learning_rates.append(learning_rate)

            # Validation phase
            self.net.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for i, data in enumerate(self.valloader):
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

            val_epoch_loss = val_loss / len(self.valloader)
            val_epoch_acc = 100 * val_correct / val_total
            self.val_losses.append(val_epoch_loss)
            self.val_accuracies.append(val_epoch_acc)

            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(
                f"Training Loss: {epoch_loss:.3f} | Training Accuracy: {epoch_acc:.2f}%"
            )
            print(
                f"Validation Loss: {val_epoch_loss:.3f} | Validation Accuracy: {val_epoch_acc:.2f}%"
            )
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            # Early stopping check
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                patience_counter = 0
                save_dir = os.path.join(os.getcwd(), "..", "..", "saved_models")
                os.makedirs(save_dir, exist_ok=True)

                filename = f"best_second_model_{self.dataset_name.lower()}.pth"
                filepath = os.path.join(save_dir, filename)
                torch.save(self.net.state_dict(), filepath)
                print(f"Saved best model with validation loss: {best_val_loss:.3f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        print("Training Completed.")

        # Save final model
        save_dir = os.path.join(os.getcwd(), "..", "..", "saved_models")
        os.makedirs(save_dir, exist_ok=True)
        filename = f"final_second_model_{self.dataset_name}.pth"
        filepath = os.path.join(save_dir, filename)
        torch.save(self.net.state_dict(), filepath)
        print(f"Final model saved to {filepath}")

        # Save the metrics to file
        file_name = f"secondmodel_{self.dataset_name}_val_{self.num_epochs}_metrics.csv"
        self.write_metrics_to_file(file_path=os.path.join(os.getcwd(), file_name))
        print(f"Metrics saved to {file_name}")

        return (
            self.train_losses,
            self.train_accuracies,
            self.val_losses,
            self.val_accuracies,
        )

    def write_metrics_to_file(self, file_path: str, without_val=False):
        """
        Write training metrics to a file.

        Parameters:
        - file_path: Path to the file where the metrics will be saved.
        """
        if without_val:
            with open(file_path, "w") as f:
                f.write("Training Loss,Training Accuracy,Learning Rate\n")
                for i in range(len(self.train_losses)):
                    f.write(
                        f"{self.train_losses[i]},{self.train_accuracies[i]},{self.learning_rates[i]}\n"
                    )
        else:
            with open(file_path, "w") as f:
                f.write(
                    "Training Loss,Training Accuracy,Validation Loss,Validation Accuracy,Learning Rate\n"
                )
                for i in range(len(self.train_losses)):
                    f.write(
                        f"{self.train_losses[i]},{self.train_accuracies[i]},{self.val_losses[i]},{self.val_accuracies[i]},{self.learning_rates[i]}\n"
                    )

    def test(self) -> tuple:
        """
        Test the neural network on the test dataset.
        """
        self.net.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        accuracy = 0.0

        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(images)
                loss = self.criterion(outputs, labels)  # criterion is CrossEntropyLoss
                test_loss += loss.item()  # sum up batch loss
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = test_loss / len(self.testloader)
        print(f"Test Loss: {avg_loss:.3f}, Accuracy: {accuracy:.3f}")
        return avg_loss, accuracy

    def predict_single_image(self, image_path: str, label_mapping=None):
        """
        Predict the class of a single image using the trained model.

        Parameters:
        - image_path: Path to the input image.
        - label_mapping: Dictionary mapping class indices to class names (optional).
        """
        # Define transformations based on the dataset
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (32, 32) if self.dataset_name.lower() == "cifar10" else (64, 64)
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(self.device)

        # Perform prediction
        self.net.eval()
        with torch.no_grad():
            outputs = self.net(image)
            _, predicted = torch.max(outputs, 1)

        # Determine the predicted class name
        if label_mapping:
            predicted_class = label_mapping.get(
                predicted.item(), f"Class {predicted.item()}"
            )
        else:
            predicted_class = f"Class {predicted.item()}"

        print(f"Predicted Class: {predicted_class}")
        return predicted_class

    def plot_training_metrics(self, file_path="secondmodel_imagenet_metrics.csv"):
        """
        Plot training metrics: loss, accuracy, learning rate, gradient norms, weight updates, and inference time.
        Uses a consistent color theme (darkred to lightpink).
        """
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        learning_rates = []

        # Read the metrics
        with open(file_path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                parts = line.strip().split(",")
                train_losses.append(float(parts[0]))
                train_accuracies.append(float(parts[1]))
                val_losses.append(float(parts[2]))
                val_accuracies.append(float(parts[3]))
                learning_rates.append(float(parts[4]))

        fig, axes = plt.subplots(1, 3, figsize=(18, 10))
        colors = [
            "darkred",
            "firebrick",
            "indianred",
            "lightcoral",
            "salmon",
            "lightpink",
        ]

        # Loss Plot
        axes[0, 0].plot(
            train_losses, label="Training Loss", marker="o", color=colors[0]
        )
        axes[0, 0].plot(
            val_losses, label="Validation Loss", marker="s", color=colors[1]
        )
        axes[0, 0].set_title(f"Training and Validation Loss ({self.dataset_name})")
        axes[0, 0].set_xlabel("Epochs")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()

        # Accuracy Plot
        axes[0, 1].plot(
            train_accuracies, label="Training Accuracy", marker="o", color=colors[2]
        )
        axes[0, 1].plot(
            val_accuracies, label="Validation Accuracy", marker="s", color=colors[3]
        )
        axes[0, 1].set_title(f"Training and Validation Accuracy ({self.dataset_name})")
        axes[0, 1].set_xlabel("Epochs")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()

        # Learning Rate Plot
        axes[0, 2].plot(
            learning_rates, label="Learning Rate", marker="o", color=colors[4]
        )
        axes[0, 2].set_title(f"Learning Rate Over Epochs {self.dataset_name}")
        axes[0, 2].set_xlabel("Epochs")
        axes[0, 2].set_ylabel("Learning Rate")
        axes[0, 2].legend()

        # Adjust layout for better visualization
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def load_model(self, model_path: str):
        """
        Load a trained model from a file.

        Parameters:
        - model_path: Path to the model file.
        """
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()
        print("Model loaded successfully.")
