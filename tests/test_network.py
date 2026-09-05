"""Unit tests for the NeuralNetwork implementation."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from neural_network import NeuralNetwork


@pytest.fixture
def xor_data():
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    y = np.array([[0, 1, 1, 0]])
    return X, y


def test_sigmoid_output_range():
    x = np.array([-1000, -1, 0, 1, 1000])
    out = NeuralNetwork.sigmoid(x)
    assert np.all(out >= 0) and np.all(out <= 1)


def test_relu_zeroes_negatives():
    x = np.array([-3, -1, 0, 2, 5])
    out = NeuralNetwork.relu(x)
    assert np.all(out >= 0)
    np.testing.assert_array_equal(out, [0, 0, 0, 2, 5])


def test_forward_output_shape(xor_data):
    X, _ = xor_data
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, seed=0)
    out = nn.forward(X)
    assert out.shape == (1, X.shape[1])


def test_forward_output_is_probability(xor_data):
    X, _ = xor_data
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, seed=0)
    out = nn.forward(X)
    assert np.all(out >= 0) and np.all(out <= 1)


def test_loss_is_nonnegative(xor_data):
    X, y = xor_data
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, seed=0)
    y_pred = nn.forward(X)
    loss = nn.compute_loss(y, y_pred)
    assert loss >= 0


def test_training_reduces_loss(xor_data):
    X, y = xor_data
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1,
                        learning_rate=0.5, seed=0)
    history = nn.train(X, y, epochs=200, verbose_every=0)
    assert history[-1] < history[0]


def test_solves_xor(xor_data):
    """With enough epochs and a good seed, the network should learn XOR."""
    X, y = xor_data
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1,
                        learning_rate=0.5, seed=1)
    nn.train(X, y, epochs=5000, verbose_every=0)
    predictions = nn.predict(X)
    np.testing.assert_array_equal(predictions, y)


def test_reproducible_with_seed():
    nn1 = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, seed=7)
    nn2 = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, seed=7)
    np.testing.assert_array_equal(nn1.W1, nn2.W1)
    np.testing.assert_array_equal(nn1.W2, nn2.W2)
