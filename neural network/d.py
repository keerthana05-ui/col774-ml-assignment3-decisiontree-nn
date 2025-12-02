import os
import sys
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt


# ReLU activation and its derivative
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


# Softmax and cross-entropy
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))


# Neural Network with ReLU
class NeuralNetworkReLU:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01, batch_size=32):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = len(hidden_sizes) + 1
        
        # Initialize weights and biases
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X):
        activations = [X]
        zs = []
        a = X
        
        for i in range(self.num_layers):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            zs.append(z)
            
            if i == self.num_layers - 1:
                a = softmax(z)
            else:
                a = relu(z)
            
            activations.append(a)
        
        return activations, zs
    
    def backward(self, y_true, activations, zs):
        grad_w = [None] * self.num_layers
        grad_b = [None] * self.num_layers
        
        # Output layer gradient (softmax + cross-entropy)
        delta = activations[-1] - y_true
        
        for i in reversed(range(self.num_layers)):
            a_prev = activations[i]
            m = a_prev.shape[0]
            
            grad_w[i] = np.dot(a_prev.T, delta) / m
            grad_b[i] = np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                delta *= relu_derivative(zs[i-1])
        
        return grad_w, grad_b
    
    def update_weights(self, grad_w, grad_b):
        for i in range(self.num_layers):
            self.weights[i] -= self.learning_rate * grad_w[i]
            self.biases[i] -= self.learning_rate * grad_b[i]
    
    def train_batch(self, X_batch, y_batch):
        activations, zs = self.forward(X_batch)
        loss = cross_entropy_loss(y_batch, activations[-1])
        grad_w, grad_b = self.backward(y_batch, activations, zs)
        self.update_weights(grad_w, grad_b)
        return loss
    
    def predict(self, X):
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)


# Data loading function with label mapping (using OpenCV)
def load_data_color_safe(data_dir, img_size=32, label_map=None):
    X_list = []
    y_list = []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    if label_map is None:
        label_map = {name: i for i, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        
        if class_name not in label_map:
            print(f"Skipping unknown class: {class_name}")
            continue
        
        label = label_map[class_name]
        
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                fpath = os.path.join(class_dir, fname)
                try:
                    img = cv2.imread(fpath)
                    if img is None:
                        print(f"Skipping unreadable file {fpath}")
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (img_size, img_size))
                    arr = img.astype(np.float32) / 255.0
                    X_list.append(arr.flatten())
                    y_list.append(label)
                except Exception:
                    print(f"Skipping file {fpath}")
    
    return np.array(X_list), np.array(y_list), label_map


def one_hot(labels, num_classes):
    return np.eye(num_classes)[labels]


def main():
    if len(sys.argv) != 4:
        print("Usage: python d.py <train_data_path> <test_data_path> <output_folder_path>")
        sys.exit(1)
    
    train_dir = sys.argv[1]
    test_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameters
    img_size = 32
    batch_size = 32
    learning_rate = 0.01
    epochs = 30
    
    print("Loading training data...")
    X_train, y_train, label_map_train = load_data_color_safe(train_dir, img_size)
    n_classes_train = len(label_map_train)
    print(f"Training data shape: {X_train.shape}, Classes: {n_classes_train}")
    
    print("Loading test data...")
    X_test, y_test, label_map_test = load_data_color_safe(test_dir, img_size)
    n_classes_test = len(label_map_test)
    print(f"Test data shape: {X_test.shape}, Test classes: {n_classes_test}")
    
    # Use maximum number of classes
    n_classes = max(n_classes_train, n_classes_test)
    print(f"Total number of classes: {n_classes}")
    
    # Remap test labels if necessary
    if label_map_test != label_map_train:
        print("Remapping test labels to match training labels...")
        combined_map = {}
        idx = 0
        for name in sorted(set(list(label_map_train.keys()) + list(label_map_test.keys()))):
            combined_map[name] = idx
            idx += 1
        n_classes = len(combined_map)
        
        # Reload with combined map
        X_train, y_train, _ = load_data_color_safe(train_dir, img_size, combined_map)
        X_test, y_test, _ = load_data_color_safe(test_dir, img_size, combined_map)
        print(f"Remapped to {n_classes} total classes")
    
    n_features = X_train.shape[1]
    y_train_oh = one_hot(y_train, n_classes)
    y_test_oh = one_hot(y_test, n_classes)
    
    print(f"One-hot encoded shapes - Train: {y_train_oh.shape}, Test: {y_test_oh.shape}")
    
    hidden_archs = [
        [512],
        [512, 256],
        [512, 256, 128],
        [512, 256, 128, 64]
    ]
    
    depths = []
    avg_f1_train_scores = []
    avg_f1_test_scores = []
    best_model = None
    best_f1 = 0
    
    for arch in hidden_archs:
        depth = len(arch)
        print(f"\n{'='*60}")
        print(f"Training with ReLU activation, depth {depth}, architecture {arch}")
        print(f"{'='*60}")
        
        nn = NeuralNetworkReLU(n_features, arch, n_classes,
                              learning_rate=learning_rate, batch_size=batch_size)
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            idx = np.arange(X_train.shape[0])
            np.random.shuffle(idx)
            X_tr = X_train[idx]
            y_tr = y_train_oh[idx]
            
            batch_losses = []
            for start in range(0, X_train.shape[0], batch_size):
                end = min(start + batch_size, X_train.shape[0])
                X_batch = X_tr[start:end]
                y_batch = y_tr[start:end]
                loss = nn.train_batch(X_batch, y_batch)
                batch_losses.append(loss)
            
            train_loss = np.mean(batch_losses)
            activations, _ = nn.forward(X_test)
            val_loss = cross_entropy_loss(y_test_oh, activations[-1])
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        train_preds = nn.predict(X_train)
        test_preds = nn.predict(X_test)
        
        precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(
            y_train, train_preds, average=None, zero_division=0)
        precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(
            y_test, test_preds, average=None, zero_division=0)
        
        avg_f1_train = np.mean(f1_train)
        avg_f1_test = np.mean(f1_test)
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        
        print(f"Train Metrics - Precision: {np.mean(precision_train):.4f}, Recall: {np.mean(recall_train):.4f}, F1: {avg_f1_train:.4f}, Accuracy: {train_acc:.4f}")
        print(f"Test Metrics - Precision: {np.mean(precision_test):.4f}, Recall: {np.mean(recall_test):.4f}, F1: {avg_f1_test:.4f}, Accuracy: {test_acc:.4f}")
        
        depths.append(depth)
        avg_f1_train_scores.append(avg_f1_train)
        avg_f1_test_scores.append(avg_f1_test)
        
        if avg_f1_test > best_f1:
            best_f1 = avg_f1_test
            best_model = nn
    
    plt.figure(figsize=(10, 6))
    plt.plot(depths, avg_f1_train_scores, marker='o', label='Train F1', linewidth=2)
    plt.plot(depths, avg_f1_test_scores, marker='s', label='Test F1', linewidth=2)
    plt.xlabel("Network Depth (Hidden Layers)", fontsize=12)
    plt.ylabel("Average F1 Score", fontsize=12)
    plt.title("Average F1 Score vs Network Depth (ReLU Activation)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'f1_vs_depth_relu.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nPlot saved to {plot_path}")
    
    if best_model is not None:
        test_predictions = best_model.predict(X_test)
        predictions_df = pd.DataFrame({'prediction': test_predictions})
        output_path = os.path.join(output_dir, 'prediction_d.csv')
        predictions_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
