# neural network/neural_network.py

import numpy as np

def sigmoid(x): #Sigmoid activation function.
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x): # Derivative of sigmoid activation function.
    
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x): #Softmax function (row-wise, numerically stable).
    
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    """
    Cross-entropy loss for one-hot targets.
    y_true: [batch_size, num_classes], one-hot encoded
    y_pred: [batch_size, num_classes], probabilities
    """
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)] + 1e-9)
    loss = np.sum(log_likelihood) / m
    return loss

def cross_entropy_softmax_gradient(y_true, y_pred):
    m = y_true.shape[0]
    return (y_pred - y_true) / m

class NeuralNetwork:
    
    def __init__(self, n_features, hidden_arch, n_classes, learning_rate=0.01, batch_size=32):
        self.n_features = n_features
        self.hidden_arch = hidden_arch
        self.n_classes = n_classes
        self.lr = learning_rate
        self.batch_size = batch_size

        # Define full layer structure: input + hidden(s) + output
        layer_sizes = [n_features] + hidden_arch + [n_classes]
        self.num_layers = len(layer_sizes) - 1
        # Weight and bias initialization
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X): #     Forward pass. Returns all activations and pre-activations.
       
        a = X
        activations = [X]
        zs = []
        for i in range(self.num_layers):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            zs.append(z)
            if i == self.num_layers - 1:
                # Output: softmax
                a = softmax(z)
            else:
                a = sigmoid(z)
            activations.append(a)
        return activations, zs

    def backward(self, X, y_true, activations, zs): #Backpropagation to compute gradients. Returns lists of gradients for weights and biases.
        grad_w = [None] * self.num_layers
        grad_b = [None] * self.num_layers

        y_pred = activations[-1]
        delta = cross_entropy_softmax_gradient(y_true, y_pred)

        for i in reversed(range(self.num_layers)):
            a_prev = activations[i]
            grad_w[i] = np.dot(a_prev.T, delta)
            grad_b[i] = np.sum(delta, axis=0, keepdims=True)
            if i != 0:
                delta = np.dot(delta, self.weights[i].T) * sigmoid_derivative(zs[i-1])
        return grad_w, grad_b

    def update_params(self, grad_w, grad_b): # Updates weights and biases with SGD.
        for i in range(self.num_layers):
            self.weights[i] -= self.lr * grad_w[i]
            self.biases[i]  -= self.lr * grad_b[i]

    def train_batch(self, X_batch, y_batch): #Train one minibatch.
        
        activations, zs = self.forward(X_batch)
        loss = cross_entropy_loss(y_batch, activations[-1])
        grad_w, grad_b = self.backward(X_batch, y_batch, activations, zs)
        self.update_params(grad_w, grad_b)
        return loss

    def predict(self, X): # Predict class indices for the input X.
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)

    def fit(self, X_train, y_train, epochs=10, X_val=None, y_val=None):  # Mini-batch SGD training loop with optional validation accuracy.
    
        n_samples = X_train.shape[0]
        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]
            batch_losses = []
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]
                loss = self.train_batch(X_batch, y_batch)
                batch_losses.append(loss)
            avg_loss = np.mean(batch_losses)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
            if X_val is not None and y_val is not None:
                val_acc = np.mean(self.predict(X_val) == np.argmax(y_val, axis=1))
                print(f"Validation Accuracy: {val_acc:.4f}")
    def save(self, path):
        
        np.savez(path, *self.weights, *self.biases)

    def load(self, path): # Load network weights and biases 
    
        npz = np.load(path)
        L = len(self.weights)
        for i in range(L):
            self.weights[i] = npz[f'arr_{i}']
            self.biases[i] = npz[f'arr_{i+L}']



