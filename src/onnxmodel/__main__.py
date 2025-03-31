# __main__.py First approach to poisoning attacks

import argparse
import sys
import torch
from pathlib import Path
from .FirstModel import FirstModel
from .SecondModel import SecondModel


def main() -> None:
    """
    Parse command-line arguments and execute the selected sub-command.
    """
    parser = argparse.ArgumentParser(
        description="Safe Lens - Analysis for Cybersecurity Vulnerabilities on AI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Sub-command: Predict
    predict_parser = subparsers.add_parser(
        "predict", help="Predict the label of an image"
    )
    predict_parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        help="Path to the image",
    )
    predict_parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["first", "second"],
        help="Model version to use (first or second)",
        default="second",
    )
    predict_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    # Sub-command: Train
    train_parser = subparsers.add_parser("train", help="Train the selected model")
    train_parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["first", "second"],
        help="Model version to train (first or second)",
        required=True,
    )
    train_parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["cifar10", "imagenet"],
        help="Dataset to use for training and testing",
        required=True,
    )
    train_parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="Number of training epochs",
        default=20,
    )
    train_parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        help="Batch size for training",
        default=64,
    )
    train_parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        help="Learning rate",
        default=0.001,
    )

    # Sub-command: Evaluate
    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate the model on the test dataset"
    )
    evaluate_parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["first", "second"],
        help="Model version to evaluate (first or second)",
        required=True,
    )

    # Sub-command: Class Accuracy
    class_acc_parser = subparsers.add_parser(
        "class_accuracy", help="Show class-wise accuracy of the model"
    )
    class_acc_parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["first", "second"],
        help="Model version to evaluate for class accuracy (first or second)",
        required=True,
    )

    # Sub-command: Visualize Data
    visualize_parser = subparsers.add_parser(
        "visualize", help="Visualize training data samples"
    )
    visualize_parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["first", "second"],
        help="Model version to use for visualization",
        default="second",
    )

    # Sub-command: Export
    export_parser = subparsers.add_parser(
        "export", help="Export the model to ONNX format"
    )
    export_parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["first", "second"],
        help="Model version to export (first or second)",
        required=True,
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Map sub-commands to functions
    try:
        if args.command == "predict":
            handle_predict(args)
        elif args.command == "train":
            handle_train(args)
        elif args.command == "evaluate":
            handle_evaluate(args)
        elif args.command == "class_accuracy":
            handle_class_accuracy(args)
        elif args.command == "visualize":
            handle_visualize(args)
        elif args.command == "export":
            handle_export(args)
        else:
            parser.print_help()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def load_model(model_name: str, dataset_name: str = "cifar10") -> torch.nn.Module:
    """
    Load the appropriate model.
    """
    if model_name == "first":
        return FirstModel(dataset_name=dataset_name)
    elif model_name == "second":
        return SecondModel(dataset_name=dataset_name)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def handle_predict(args):
    """
    Predict the label of a given image using the selected model.
    """
    model = load_model(
        args.model,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    prediction = model.predict(
        image_path
    )  # Assuming predict method exists in the model class
    print(f"Predicted label: {prediction}")
    if args.verbose:
        print(
            f"Details: {prediction.details()}"
        )  # Assuming prediction includes detailed info


def handle_train(args):
    """
    Train the selected model with specified parameters.
    """
    model = load_model(args.model)
    model.train()
    print(f"Training completed for {args.model} model.")


def handle_evaluate(args):
    """
    Evaluate the selected model on the test dataset.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device=device)
    accuracy = model.evaluate()
    print(f"Evaluation completed. Accuracy: {accuracy:.2f}%")


def handle_class_accuracy(args):
    """
    Display class-wise accuracy for the selected model.
    """
    model = load_model(args.model)
    model.class_accuracy()


def handle_visualize(args):
    """
    Visualize data samples from the training set.
    """
    model = load_model(args.model)
    model.visualize_data()


def handle_export(args):
    """
    Export the selected model to ONNX format.
    """
    model = load_model(args.model)
    model.export_onnx()
    print(f"Model exported to ONNX format for {args.model}.")


if __name__ == "__main__":
    main()
