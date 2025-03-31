import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from cleverhans.torch.attacks import fast_gradient_method
from onnxmodel.SecondModel import SecondModel
import torch.nn.functional as F
import random
import copy


class PoisonAttack:
    def __init__(self, model=None, poison_ratio=0.1, target_label=None) -> None:
        """
        Initialize poisoning attack.

        Args:
            model: The model to attack (if None, creates new SecondModel)
            poison_ratio: Percentage of training data to poison (default 0.1 = 10%)
            target_label: Label to flip to (if None, randomly assigns new labels)
        """
        # check if the model is a SecondModel, otherwise error
        if model is not None and not isinstance(model, SecondModel):
            raise ValueError("Model must be an instance of SecondModel")
        else:
            self.model = model
        self.poison_ratio = poison_ratio
        self.target_label = target_label
        self.model.net.eval()

    def create_poisoned_dataset(self, dataset):
        """
        Create a poisoned version of the dataset with flipped labels.

        Args:
            dataset: Dataset to poison (can be original dataset or Subset)

        Returns:
            Poisoned dataset
        """
        num_samples = len(dataset)

        # Handle Subset objects by accessing their underlying dataset
        original_dataset = (
            dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
        )

        # Determine number of classes based on the dataset type
        if isinstance(original_dataset, torchvision.datasets.CIFAR10):
            num_classes = 10
        elif hasattr(original_dataset, "classes"):
            num_classes = len(original_dataset.classes)
        else:
            # Try to infer from the model's output layer
            num_classes = self.model.net.fc3.out_features

        num_poison = int(num_samples * self.poison_ratio)

        # Select indices to poison
        indices_to_poison = random.sample(range(num_samples), num_poison)

        # Create a copy of the dataset
        poisoned_dataset = copy.deepcopy(dataset)

        # If it's a Subset, we need to map the indices to the original dataset
        if isinstance(poisoned_dataset, torch.utils.data.Subset):
            for idx in indices_to_poison:
                # Get the original index in the full dataset
                original_idx = poisoned_dataset.indices[idx]

                # Get current label
                if hasattr(original_dataset, "targets"):
                    current_label = original_dataset.targets[original_idx]
                elif hasattr(original_dataset, "labels"):
                    current_label = original_dataset.labels[original_idx]
                else:
                    _, current_label = original_dataset[original_idx]

                # Generate new label
                if self.target_label is None:
                    new_label = random.choice(
                        [l for l in range(num_classes) if l != current_label]
                    )
                else:
                    new_label = self.target_label

                # Update the label in the original dataset
                if hasattr(original_dataset, "targets"):
                    original_dataset.targets[original_idx] = new_label
                elif hasattr(original_dataset, "labels"):
                    original_dataset.labels[original_idx] = new_label
                else:
                    raise NotImplementedError(
                        "Dataset format not supported for poisoning"
                    )
        else:
            # Original implementation for non-Subset datasets
            for idx in indices_to_poison:
                if hasattr(poisoned_dataset, "targets"):
                    current_label = poisoned_dataset.targets[idx]
                elif hasattr(poisoned_dataset, "labels"):
                    current_label = poisoned_dataset.labels[idx]
                else:
                    _, current_label = poisoned_dataset[idx]

                if self.target_label is None:
                    new_label = random.choice(
                        [l for l in range(num_classes) if l != current_label]
                    )
                else:
                    new_label = self.target_label

                if hasattr(poisoned_dataset, "targets"):
                    poisoned_dataset.targets[idx] = new_label
                elif hasattr(poisoned_dataset, "labels"):
                    poisoned_dataset.labels[idx] = new_label
                else:
                    raise NotImplementedError(
                        "Dataset format not supported for poisoning"
                    )

        print(
            f"Created poisoned dataset with {num_poison} poisoned samples ({self.poison_ratio*100:.1f}%)"
        )
        return poisoned_dataset


class PoisonedDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            data = self.transform(data)
        return data, label

    def attack(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        testset = torchvision.datasets.CIFAR10(
            root="../data", train=False, download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

        # Generate adversarial examples using FGSM
        for inputs, labels in testloader:
            inputs.requires_grad = True  # Enable gradient tracking for inputs
            outputs = self.model(inputs)  # Forward pass through the model
            loss = F.cross_entropy(outputs, labels)  # Calculate loss
            self.model.zero_grad()  # Zero gradients before backward pass
            loss.backward()  # Backward pass to calculate gradients

            # Generate adversarial example
            eps = 0.1  # Perturbation amount
            adv_inputs = fast_gradient_method(
                self.model, inputs, eps, np.inf
            )  # FGSM attack

            # Check the model's prediction on adversarial example
            adv_outputs = self.model(adv_inputs)
            _, predicted = torch.max(adv_outputs.data, 1)

            print(
                f"Original label: {labels.item()}, Adversarial label: {predicted.item()}"
            )
