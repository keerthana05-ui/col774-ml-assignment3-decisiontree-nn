import os
import sys
import numpy as np
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import pandas as pd


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
                except Exception:
                    print(f"Warning: Skipping file due to error: {fpath}")
                    continue
    return np.array(X_list), np.array(y_list)


def main(train_dir, test_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_size = 32
    n_features = img_size * img_size * 3
    n_classes = 36
    batch_size = 32
    learning_rate_init = 0.01
    max_iter = 30

    print("Loading training data...")
    X_train, y_train = load_data_color_safe(train_dir, n_classes, img_size)
    print(f"Loaded {X_train.shape[0]} training samples.")

    print("Loading test data...")
    X_test, y_test = load_data_color_safe(test_dir, n_classes, img_size)
    print(f"Loaded {X_test.shape[0]} test samples.")

    hidden_archs = [
        (512,),
        (512, 256),
        (512, 256, 128),
        (512, 256, 128, 64)
    ]

    avg_f1_scores = []
    depths = []

    for arch in hidden_archs:
        depth = len(arch)
        print(f"\nTraining sklearn MLPClassifier with depth {depth}, hidden sizes: {arch}")
        clf = MLPClassifier(
            hidden_layer_sizes=arch,
            activation='relu',
            solver='sgd',
            alpha=0,
            batch_size=batch_size,
            learning_rate='constant',
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            tol=1e-4,
            n_iter_no_change=5,
            verbose=True,
            random_state=42,
        )

        clf.fit(X_train, y_train)
        train_preds = clf.predict(X_train)
        test_preds = clf.predict(X_test)

        precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(
            y_train, train_preds, average=None, zero_division=0)
        precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(
            y_test, test_preds, average=None, zero_division=0)

        avg_f1_train = np.mean(f1_train)
        avg_f1_test = np.mean(f1_test)
        avg_f1_scores.append(avg_f1_test)

        print(f"Average Train F1: {avg_f1_train:.4f}")
        print(f"Average Test F1: {avg_f1_test:.4f}")
        print("Test per-class precision, recall, F1:")
        for i in range(n_classes):
            print(f" Class {i}: P={precision_test[i]:.3f} R={recall_test[i]:.3f} F1={f1_test[i]:.3f}")

        depths.append(depth)

    best_index = np.argmax(avg_f1_scores)
    best_arch = hidden_archs[best_index]

    print(f"\nRetraining best model with hidden sizes {best_arch} for final predictions...")
    clf = MLPClassifier(
        hidden_layer_sizes=best_arch,
        activation='relu',
        solver='sgd',
        alpha=0,
        batch_size=batch_size,
        learning_rate='constant',
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        tol=1e-4,
        n_iter_no_change=5,
        verbose=True,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    final_test_preds = clf.predict(X_test)

    output_file = os.path.join(output_dir, 'prediction_e.csv')
    pd.DataFrame({'prediction': final_test_preds}).to_csv(output_file, index=False)
    print(f"Test predictions saved to {output_file}")

    plt.plot(depths, avg_f1_scores, marker='o')
    plt.xlabel('Network Depth (Number of Hidden Layers)')
    plt.ylabel('Average Test F1 Score')
    plt.title('Average Test F1 Score vs Network Depth (MLPClassifier)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'f1_vs_network_depth_sklearn.png'))
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python e.py <train_data_path> <test_data_path> <output_folder_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
