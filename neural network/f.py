import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from neural_network import NeuralNetworkReLU
import pandas as pd


digits_train_dir = 'train'
digits_test_dir = 'test'
consonant_model_path = "consonant_mlp.npz"

IMG_SIZE = 32
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.01
HIDDEN_SIZES = [512, 256, 128, 64]
np.random.seed(42)


def load_data_color_safe(data_dir, img_size=32):
    X_list = []
    y_list = []
    class_names = sorted(os.listdir(data_dir))
    for idx, name in enumerate(class_names):
        class_dir = os.path.join(data_dir, name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('png','jpg','jpeg')):
                fpath = os.path.join(class_dir, fname)
                try:
                    img = cv2.imread(fpath)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (img_size, img_size))
                    arr = img.astype(np.float32) / 255.0
                    X_list.append(arr.flatten())
                    y_list.append(idx)
                except Exception:
                    continue
    return np.array(X_list), np.array(y_list), class_names


def one_hot(labels, num_classes): 
    return np.eye(num_classes)[labels]


def main(train_dir, test_dir, output_dir):
    global digits_train_dir, digits_test_dir, consonant_model_path
    digits_train_dir = train_dir
    digits_test_dir = test_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading digits train...")
    X_train, y_train, digits_classes = load_data_color_safe(digits_train_dir, IMG_SIZE)
    print(f"Loaded {X_train.shape[0]} train samples with {len(digits_classes)} digit classes")
    print("Loading digits test...")
    X_test, y_test, _ = load_data_color_safe(digits_test_dir, IMG_SIZE)
    print(f"Loaded {X_test.shape[0]} test samples")

    num_classes = max(np.max(y_train), np.max(y_test)) + 1
    y_train_oh = one_hot(y_train, num_classes)
    y_test_oh = one_hot(y_test, num_classes)

    # Train from scratch on digits
    print("\nTraining from Scratch on Digits Subset")
    mlp_scratch = NeuralNetworkReLU([IMG_SIZE*IMG_SIZE*3]+HIDDEN_SIZES+[num_classes], learning_rate=LEARNING_RATE)
    train_f1s, test_f1s = [], []
    for epoch in range(EPOCHS):
        idx = np.arange(X_train.shape[0])
        np.random.shuffle(idx)
        X_train_s, y_train_oh_s = X_train[idx], y_train_oh[idx]
        for start in range(0, X_train.shape[0], BATCH_SIZE):
            end = start + BATCH_SIZE
            mlp_scratch.train_batch(X_train_s[start:end], y_train_oh_s[start:end])
        train_pred = mlp_scratch.predict(X_train)
        test_pred = mlp_scratch.predict(X_test)
        train_f1 = f1_score(y_train, train_pred, average='macro', zero_division=0)
        test_f1 = f1_score(y_test, test_pred, average='macro', zero_division=0)
        train_f1s.append(train_f1)
        test_f1s.append(test_f1)
        print(f"Epoch {epoch+1}: Train F1={train_f1:.4f} Test F1={test_f1:.4f}")

    output_file_scratch = os.path.join(output_dir, 'prediction_f_scratch.csv')
    pd.DataFrame({'prediction': test_pred}).to_csv(output_file_scratch, index=False)
    print(f"Saved scratch test predictions to {output_file_scratch}")

    # Transfer learning fine-tuning
    print("\nTransfer Learning: Fine-tune from Consonant Model")
    mlp_transfer = NeuralNetworkReLU([IMG_SIZE*IMG_SIZE*3]+HIDDEN_SIZES+[num_classes], learning_rate=LEARNING_RATE)
    mlp_consonant = NeuralNetworkReLU([IMG_SIZE*IMG_SIZE*3]+HIDDEN_SIZES+[36], learning_rate=LEARNING_RATE)

    if os.path.exists(consonant_model_path):
        mlp_consonant.load(consonant_model_path)
        mlp_transfer.set_weights_from(mlp_consonant, num_classes)
    else:
        print(f"Consonant model weights file '{consonant_model_path}' not found.")
        print("Transfer learning will proceed with random initialization for all layers.")

    transfer_train_f1s, transfer_test_f1s = [], []
    for epoch in range(EPOCHS):
        idx = np.arange(X_train.shape[0])
        np.random.shuffle(idx)
        X_train_s, y_train_oh_s = X_train[idx], y_train_oh[idx]
        for start in range(0, X_train.shape[0], BATCH_SIZE):
            end = start + BATCH_SIZE
            mlp_transfer.train_batch(X_train_s[start:end], y_train_oh_s[start:end])
        train_pred = mlp_transfer.predict(X_train)
        test_pred = mlp_transfer.predict(X_test)
        train_f1 = f1_score(y_train, train_pred, average='macro', zero_division=0)
        test_f1 = f1_score(y_test, test_pred, average='macro', zero_division=0)
        transfer_train_f1s.append(train_f1)
        transfer_test_f1s.append(test_f1)
        print(f"Epoch {epoch+1}: Transfer Train F1={train_f1:.4f} Transfer Test F1={test_f1:.4f}")

    output_file_transfer = os.path.join(output_dir, 'prediction_f_transfer.csv')
    pd.DataFrame({'prediction': test_pred}).to_csv(output_file_transfer, index=False)
    print(f"Saved transfer test predictions to {output_file_transfer}")

    plt.plot(range(1, EPOCHS+1), test_f1s, label='Train-from-Scratch Test F1')
    plt.plot(range(1, EPOCHS+1), transfer_test_f1s, label='Transfer Learning Test F1')
    plt.xlabel('Epoch')
    plt.ylabel('Test Macro F1 Score')
    plt.title('Digits Subset Test F1: Train-from-Scratch vs Transfer Learning')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "transfer_digits_f1.png"))
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python f.py <train_data_path> <test_data_path> <output_folder_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
