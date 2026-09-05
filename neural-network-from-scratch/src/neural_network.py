"""
Feedforward Neural Network from Scratch (NumPy only)
=====================================================

A minimal implementation of a feedforward neural network with one hidden
layer, trained via manual backpropagation. Built without any deep learning
framework (no PyTorch/TensorFlow) to understand the mechanics of forward
propagation, backpropagation, and gradient descent at the matrix level.

Architecture:
    Input layer -> Hidden layer (ReLU) -> Output layer (Sigmoid)

Loss:
    Binary Cross-Entropy

Example:
    Solving XOR, a classic non-linearly-separable problem that requires
    a hidden layer to solve.
"""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """A simple 2-layer feedforward neural network (1 hidden layer + output).

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_size : int
        Number of neurons in the hidden layer.
    output_size : int
        Number of output neurons.
    learning_rate : float
        Step size used during gradient descent updates.
    seed : int, optional
        Random seed for reproducible weight initialization.
    """

    def __init__(self, input_size, hidden_size, output_size,
                 learning_rate=0.1, seed=42):
        self.learning_rate = learning_rate
        rng = np.random.default_rng(seed)

        # He/Xavier-style small random initialization instead of hardcoded values
        self.W1 = rng.normal(0, 1, size=(hidden_size, input_size)) * np.sqrt(1 / input_size)
        self.b1 = np.zeros((hidden_size, 1))

        self.W2 = rng.normal(0, 1, size=(output_size, hidden_size)) * np.sqrt(1 / hidden_size)
        self.b2 = np.zeros((output_size, 1))

        self.loss_history = []

    # ------------------------------------------------------------------
    # Activation functions
    # ------------------------------------------------------------------
    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)  # prevent overflow
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, X):
        """Run a forward pass.

        Parameters
        ----------
        X : ndarray of shape (input_size, n_samples)

        Returns
        -------
        A2 : ndarray of shape (output_size, n_samples)
            Network predictions.
        """
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    @staticmethod
    def compute_loss(y_true, y_pred):
        """Binary cross-entropy loss, averaged over all samples."""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    def backward(self, X, y_true):
        """Compute gradients and update weights via gradient descent.

        Parameters
        ----------
        X : ndarray of shape (input_size, n_samples)
        y_true : ndarray of shape (output_size, n_samples)
        """
        n_samples = X.shape[1]

        # Output layer gradients (sigmoid + BCE derivative simplifies to A2 - y)
        dZ2 = self.A2 - y_true
        dW2 = np.dot(dZ2, self.A1.T) / n_samples
        db2 = np.sum(dZ2, axis=1, keepdims=True) / n_samples

        # Hidden layer gradients
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * (self.Z1 > 0)  # ReLU derivative
        dW1 = np.dot(dZ1, X.T) / n_samples
        db1 = np.sum(dZ1, axis=1, keepdims=True) / n_samples

        # Gradient descent update
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self, X, y, epochs=5000, verbose_every=500):
        """Train the network for a fixed number of epochs (full-batch).

        Parameters
        ----------
        X : ndarray of shape (input_size, n_samples)
        y : ndarray of shape (output_size, n_samples)
        epochs : int
        verbose_every : int
            Print progress every N epochs (0 to disable).
        """
        self.loss_history = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            self.backward(X, y)

            if verbose_every and (epoch + 1) % verbose_every == 0:
                print(f"Epoch {epoch + 1:5d} | Loss: {loss:.6f}")

        return self.loss_history

    def predict(self, X, threshold=0.5):
        """Return binary predictions (0/1) for input X."""
        probs = self.forward(X)
        return (probs > threshold).astype(int)

    def plot_loss(self, save_path=None, show=True):
        """Plot the training loss curve.

        Parameters
        ----------
        save_path : str, optional
            If given, saves the figure to this path (e.g. 'assets/loss_plot.png').
        show : bool
            Whether to display the plot interactively.
        """
        plt.figure(figsize=(7, 4))
        plt.plot(self.loss_history)
        plt.title("Training Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross-Entropy Loss")
        plt.grid(alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()


def main():
    # XOR dataset: not linearly separable, requires a hidden layer to solve.
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])          # shape: (2 features, 4 samples)
    y = np.array([[0, 1, 1, 0]])          # shape: (1 output, 4 samples)

    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1,
                        learning_rate=0.5, seed=1)

    print("Training on XOR problem...\n")
    nn.train(X, y, epochs=5000, verbose_every=500)

    predictions = nn.forward(X)
    print("\nFinal predictions vs targets:")
    for i in range(X.shape[1]):
        print(f"  Input: {X[:, i]} | Predicted: {predictions[0, i]:.4f} | Target: {y[0, i]}")

    nn.plot_loss(save_path="assets/loss_plot.png", show=False)
    print("\nLoss plot saved to assets/loss_plot.png")


if __name__ == "__main__":
    main()
