import sys
import os
import pandas as pd
from decision_tree import DecisionTree

one_hot_cols = ['team', 'opp', 'host', 'month']

def load_and_onehot(file):
    df = pd.read_csv(file)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    y = df['result'] if 'result' in df.columns else None
    X = df.drop(columns=['result']) if 'result' in df.columns else df
    X = pd.get_dummies(X, columns=one_hot_cols)
    return X, y

def main():
    if len(sys.argv) != 5:
        print("Usage: python b.py <train_data_path> <val_data_path> <test_data_path> <output_path>")
        sys.exit(1)

    train_path = sys.argv[1]
    val_path = sys.argv[2]
    test_path = sys.argv[3]
    output_path = sys.argv[4]

    X_train, y_train = load_and_onehot(train_path)
    X_val, y_val = load_and_onehot(val_path)
    X_test, _ = load_and_onehot(test_path)

    # Align columns
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Train and select best depth
    depths = [15, 25, 35, 45]
    best_val_acc = 0
    best_tree = None

    for d in depths:
        tree = DecisionTree(max_depth=d)
        tree.fit(X_train, y_train)
        val_pred = tree.predict(X_val)
        val_acc = (val_pred == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_tree = tree

    test_pred = best_tree.predict(X_test) 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create output directory if needed
    pd.DataFrame({'result': test_pred}).to_csv(output_path, index=False)  # Save results with 'result' column

if __name__ == '__main__':
    main()
