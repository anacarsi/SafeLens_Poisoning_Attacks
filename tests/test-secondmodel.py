import pytest
from ..src.onnxmodel import SecondModel
from unittest.mock import patch


@pytest.fixture
def second_model():
    return SecondModel(batch_size=2, learning_rate=0.001, num_epochs=1)


def test_second_model_train(second_model):
    second_model.build_model()
    second_model.train()
    assert hasattr(
        second_model, "net"
    ), "SecondModel should have a 'net' attribute after build_model"


def test_second_model_accuracy(second_model):
    second_model.build_model()
    accuracy = second_model.test()
    assert (
        accuracy is not None
    ), "SecondModel test method should return an accuracy value"


def test_train_with_mocked_data(second_model):
    second_model.build_model()
    with patch.object(
        second_model, "_prepare_data", return_value=None
    ):  # Mock data loading
        second_model.train()
    assert hasattr(second_model, "net")
