import numpy as np
from src import logger


class SingleLayerNN:
    """
    Single-layer Neural Network (Logistic Regression)
    N inputs -> 1 output
    """

    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.weights = None
        self.bias = None
        self.history = {'loss': [], 'accuracy': []}
        self.logger = logger.bind(component="SingleLayerNN")

    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def initialize_weights(self):
        """Initialize weights with small random values"""
        np.random.seed(42)
        self.weights = np.random.randn(self.input_dim) * 0.01
        self.bias = 0.0
        self.logger.debug("weights_initialized", input_dim=self.input_dim)

    def forward(self, X):
        """Forward pass"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def compute_loss(self, y_pred, y_true):
        """Binary cross-entropy loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def compute_accuracy(self, y_pred, y_true):
        """Compute classification accuracy"""
        predictions = (y_pred >= 0.5).astype(int)
        return np.mean(predictions == y_true)

    def train(
        self, X, y,
        X_val=None, y_val=None,
        learning_rate=0.01,
        epochs=1000,
        batch_size=32,
        verbose=True,
        verbose_every=100,
        patience=0,
        l2_lambda=0.0
    ):
        """
        Train the neural network with optional early stopping and L2 regularization

        Args:
            X: Training data (n_samples, input_dim)
            y: Labels (n_samples,)
            X_val: Validation data (optional, for early stopping)
            y_val: Validation labels (optional, for early stopping)
            learning_rate: Learning rate for gradient descent
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
            verbose: Whether to print training progress
            verbose_every: Print progress every N epochs
            patience: Number of epochs to wait for improvement before stopping (0 = disabled)
            l2_lambda: L2 regularization strength (weight decay). 0 = disabled.
        """
        if self.weights is None:
            self.initialize_weights()

        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size

        # Early stopping variables
        best_val_acc = 0.0
        best_epoch = 0
        best_weights = None
        best_bias = None
        epochs_without_improvement = 0
        use_early_stopping = patience > 0 and X_val is not None and y_val is not None

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0

            # Mini-batch gradient descent
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Compute loss
                loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += loss

                # Backward pass (gradient computation)
                error = y_pred - y_batch
                dw = np.dot(X_batch.T, error) / len(X_batch)
                db = np.mean(error)

                # Add L2 regularization gradient (weight decay)
                if l2_lambda > 0:
                    dw += l2_lambda * self.weights

                # Update weights
                self.weights -= learning_rate * dw
                self.bias -= learning_rate * db

            # Compute metrics for full dataset
            y_pred_full = self.forward(X)
            avg_loss = epoch_loss / n_batches
            accuracy = self.compute_accuracy(y_pred_full, y)

            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(accuracy)

            # Compute validation metrics if provided
            if use_early_stopping:
                y_val_pred = self.forward(X_val)
                val_acc = self.compute_accuracy(y_val_pred, y_val)
                val_loss = self.compute_loss(y_val_pred, y_val)

                if 'val_loss' not in self.history:
                    self.history['val_loss'] = []
                    self.history['val_acc'] = []
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # Early stopping check (>= so plateaus reset the patience counter)
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch + 1
                    best_weights = self.weights.copy()
                    best_bias = self.bias
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            if verbose and verbose_every > 0 and (epoch + 1) % verbose_every == 0:
                log_kwargs = {
                    'epoch': epoch + 1,
                    'total_epochs': epochs,
                    'loss': round(avg_loss, 4),
                    'accuracy': round(accuracy, 4)
                }
                if use_early_stopping:
                    log_kwargs['val_acc'] = round(val_acc, 4)
                    log_kwargs['best_val_acc'] = round(best_val_acc, 4)
                    log_kwargs['epochs_no_improve'] = epochs_without_improvement
                self.logger.info("training_progress", **log_kwargs)

            # Early stopping
            if use_early_stopping and epochs_without_improvement >= patience:
                self.logger.info("early_stopping",
                               stopped_at_epoch=epoch + 1,
                               best_epoch=best_epoch,
                               best_val_acc=round(best_val_acc, 4),
                               patience=patience)
                break

        # Restore best weights if early stopping was used
        if use_early_stopping and best_weights is not None:
            self.weights = best_weights
            self.bias = best_bias
            self.history['best_epoch'] = best_epoch
            self.history['best_val_acc'] = best_val_acc
            self.logger.info("restored_best_model",
                           best_epoch=best_epoch,
                           best_val_acc=round(best_val_acc, 4))

    def predict(self, X):
        """Make predictions"""
        y_pred = self.forward(X)
        return (y_pred >= 0.5).astype(int)

    def get_weights(self):
        """Return the learned weights"""
        return self.weights

