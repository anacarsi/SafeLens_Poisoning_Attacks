import pytest
from ..src.onnxmodel import FirstModel
from unittest.mock import patch


@pytest.fixture
def first_model():
    return FirstModel(batch_size=2, learning_rate=0.001, num_epochs=1)


def test_train_with_mocked_data(first_model):
    first_model.build_model()
    with patch.object(
        first_model, "_prepare_data", return_value=None
    ):  # Mock data loading
        first_model.train()
    assert hasattr(first_model, "net")


def test_first_model_train(first_model):
    first_model.build_model()  # Ensures the model and optimizer are initialized
    first_model.train()  # Check if training completes without errors
    assert hasattr(
        first_model, "net"
    ), "FirstModel should have a 'net' attribute after build_model"


def test_first_model_accuracy(first_model):
    first_model.build_model()
    accuracy = first_model.test()
    assert (
        accuracy is not None
    ), "FirstModel test method should return an accuracy value"
