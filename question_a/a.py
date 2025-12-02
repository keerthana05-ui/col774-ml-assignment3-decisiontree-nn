import sys
import pandas as pd
import matplotlib.pyplot as plt
from decision_tree import DecisionTree, accuracy
import os

def main():
    if len(sys.argv) != 5:
        print("Usage: python a.py <train_data_path> <val_data_path> <test_data_path> <output_folder_path>")
        sys.exit(1)

    train_path = sys.argv[1]
    val_path = sys.argv[2]
    test_path = sys.argv[3]
    output_folder_path = sys.argv[4]

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    features = [col for col in train_df.columns if col not in ['result', 'Unnamed']]
    X_train = train_df[features]
    y_train = train_df['result']
    X_valid = valid_df[features]
    y_valid = valid_df['result']
    if "result" in test_df.columns:
        X_test = test_df[features]
    else:
        X_test = test_df[features]

    depths = [5, 10, 15, 20]
    train_acc = []
    valid_acc = []

    for d in depths:
        print(f"\nTraining tree of max depth {d}")
        tree = DecisionTree(max_depth=d)
        tree.fit(X_train, y_train)
        train_pred = tree.predict(X_train)
        valid_pred = tree.predict(X_valid)
        ta = accuracy(y_train, train_pred)
        va = accuracy(y_valid, valid_pred)
        print(f"Depth {d}: Train Acc = {ta:.4f}, Validation Acc = {va:.4f}")
        train_acc.append(ta)
        valid_acc.append(va)

    plt.plot(depths, train_acc, marker='o', label='Train Accuracy')
    plt.plot(depths, valid_acc, marker='o', label='Validation Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Accuracy vs Depth')
    plt.legend()
    plt.savefig('tree_accuracy.png')
    plt.show()

    # Use the last fitted tree for test prediction
    test_pred = tree.predict(X_test)
    output_folder = os.path.dirname(output_folder_path)  # Ensure output  exists
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pd.DataFrame({'result': test_pred}).to_csv(output_folder_path, index=False)

if __name__ == '__main__':
    main()
