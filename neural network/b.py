import os
import sys
import numpy as np
import cv2
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork  

def load_data_color_safe(data_dir, n_classes=36, img_size=32):
    X_list = []
    y_list = []
    class_names = sorted(os.listdir(data_dir))
    label_map = {name: i for i, name in enumerate(class_names)}
    for name in class_names:
        class_dir = os.path.join(data_dir, name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                fpath = os.path.join(class_dir, fname)
                try:
                    img = cv2.imread(fpath)
                    if img is None:
                        print(f"Warning: Skipping unreadable image: {fpath}")
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (img_size, img_size))
                    arr = img.astype(np.float32) / 255.0
                    X_list.append(arr.flatten())
                    y_list.append(label_map[name])
                except Exception as e:
                    print(f"Warning: Skipping file due to error: {fpath}")
                    continue
    return np.array(X_list), np.array(y_list)

def one_hot(labels, num_classes):
    return np.eye(num_classes)[labels]

def main(train_dir, test_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_classes = 36
    img_size = 32
    n_features = img_size * img_size * 3
    batch_size = 32
    learning_rate = 0.01
    epochs = 30

    print("Loading training data...")
    X_train, y_train = load_data_color_safe(train_dir, n_classes, img_size)
    print(f"Loaded {X_train.shape[0]} training samples.")

    print("Loading test data...")
    X_test, y_test = load_data_color_safe(test_dir, n_classes, img_size)
    print(f"Loaded {X_test.shape[0]} test samples.")

    y_train_oh = one_hot(y_train, n_classes)
    y_test_oh = one_hot(y_test, n_classes)

    hidden_units_options = [1, 5, 10, 50, 100]
    avg_f1_scores = []
    best_test_preds = None
    best_f1_score = -1

    for hidden_units in hidden_units_options:
        print(f"\nTraining: Single hidden layer units = {hidden_units}")
        nn = NeuralNetwork(n_features, [hidden_units], n_classes,
                           learning_rate=learning_rate, batch_size=batch_size)

        best_val_loss = np.inf
        patience = 5
        patience_counter = 0

        for epoch in range(epochs):
            idx = np.arange(X_train.shape[0])
            np.random.shuffle(idx)
            X_train_s = X_train[idx]
            y_train_s = y_train_oh[idx]

            losses = []
            for start in range(0, X_train.shape[0], batch_size):
                end = start + batch_size
                X_batch = X_train_s[start:end]
                y_batch = y_train_s[start:end]
                loss = nn.train_batch(X_batch, y_batch)
                losses.append(loss)
            train_loss = np.mean(losses)

            activations, _ = nn.forward(X_test)
            val_loss = np.mean([-np.log(activations[-1][i, y_test[i]] + 1e-9) for i in range(len(y_test))])

            print(f" Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        train_preds = nn.predict(X_train)
        test_preds = nn.predict(X_test)

        precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(
            y_train, train_preds, average=None, zero_division=0)
        precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(
            y_test, test_preds, average=None, zero_division=0)

        avg_f1_train = np.mean(f1_train)
        avg_f1_test = np.mean(f1_test)
        avg_f1_scores.append(avg_f1_test)

        print(f" Average Train F1: {avg_f1_train:.4f}")
        print(f" Average Test F1: {avg_f1_test:.4f}")
        print("Test per-class precision, recall, F1:")
        for i in range(n_classes):
            print(f" Class {i}: P={precision_test[i]:.3f} R={recall_test[i]:.3f} F1={f1_test[i]:.3f}")

        if avg_f1_test > best_f1_score:
            best_f1_score = avg_f1_test
            best_test_preds = test_preds

    output_file = os.path.join(output_dir, 'prediction_b.csv')  # Save best predictions
    import pandas as pd
    pd.DataFrame({'prediction': best_test_preds}).to_csv(output_file, index=False)
    print(f"Saved best test predictions to {output_file}")

    plt.plot(hidden_units_options, avg_f1_scores, marker='o')
    plt.xlabel('Hidden layer units')
    plt.ylabel('Average Test F1 score')
    plt.title('F1 Score vs Hidden Layer Units')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'f1_vs_hidden_units.png'))
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python b.py <train_data_path> <test_data_path> <output_folder_path>")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
