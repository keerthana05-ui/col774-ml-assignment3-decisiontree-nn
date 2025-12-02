import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from decision_tree import DecisionTree, accuracy

one_hot_cols = ['team', 'opp', 'host', 'month']

def load_and_onehot(file):
    df = pd.read_csv(file)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    y = df['result'] if 'result' in df.columns else None
    X = df.drop(columns=['result']) if 'result' in df.columns else df
    X = pd.get_dummies(X, columns=one_hot_cols)
    return X, y

def count_nodes(node):
    if node.is_leaf:
        return 1
    return 1 + sum(count_nodes(child) for child in node.children.values())

def get_non_leaf_nodes(node, nodes=None):
    if nodes is None:
        nodes = []
    if not node.is_leaf:
        nodes.append(node)
        for child in node.children.values():
            get_non_leaf_nodes(child, nodes)
    return nodes

def make_leaf(node, y_subset):
    node.is_leaf = True
    node.prediction = y_subset.value_counts().idxmax()
    node.split_attr = None
    node.split_val = None
    node.children = {}

def post_prune(tree, X_val, y_val, X_train, y_train, X_test, y_test,
               max_prune_iters=15, delta_threshold=1e-4):
    print("Starting Post-pruning...")
    current_acc_val = accuracy(y_val, tree.predict(X_val))
    current_acc_train = accuracy(y_train, tree.predict(X_train))
    current_acc_test = accuracy(y_test, tree.predict(X_test))
    accuracies_train = [current_acc_train]
    accuracies_val = [current_acc_val]
    accuracies_test = [current_acc_test]
    node_counts = [count_nodes(tree.root)]

    for iteration in range(max_prune_iters):
        nodes = get_non_leaf_nodes(tree.root)
        best_acc_increase = 0
        node_to_prune = None

        for node in nodes:
            backup = (node.is_leaf, node.prediction, node.split_attr, node.split_val, node.children)
            make_leaf(node, y_val)
            val_acc = accuracy(y_val, tree.predict(X_val))
            if val_acc - current_acc_val > best_acc_increase:
                best_acc_increase = val_acc - current_acc_val
                node_to_prune = node
            node.is_leaf, node.prediction, node.split_attr, node.split_val, node.children = backup

        if node_to_prune is None or best_acc_increase < delta_threshold:
            print(f"No significant improvement at iteration {iteration}, stopping pruning.")
            break

        make_leaf(node_to_prune, y_val)
        current_acc_val += best_acc_increase
        current_acc_train = accuracy(y_train, tree.predict(X_train))
        current_acc_test = accuracy(y_test, tree.predict(X_test))

        node_counts.append(count_nodes(tree.root))
        accuracies_train.append(current_acc_train)
        accuracies_val.append(current_acc_val)
        accuracies_test.append(current_acc_test)

        print(f"Iteration {iteration}: Validation accuracy = {current_acc_val:.4f}, Node count = {node_counts[-1]}")

    print("Post-pruning finished.")
    return node_counts, accuracies_train, accuracies_val, accuracies_test

def run_all_curves(train_path, val_path, test_path, output_folder):
    print("Loading data...")
    X_train, y_train = load_and_onehot(train_path)
    X_val, y_val = load_and_onehot(val_path)
    X_test, y_test = load_and_onehot(test_path)

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    depths = [15, 25, 35, 45]
    plt.figure(figsize=(14, 15))

    for i, max_depth in enumerate(depths):
        print(f"Training DecisionTree with max_depth={max_depth}...")
        tree = DecisionTree(max_depth=max_depth)
        tree.fit(X_train, y_train)
        node_counts, train_accs, val_accs, test_accs = post_prune(
            tree, X_val, y_val, X_train, y_train, X_test, y_test,
            max_prune_iters=15, delta_threshold=1e-4
        )
        plt.subplot(len(depths), 1, i + 1)
        plt.plot(node_counts, train_accs, label='Train')
        plt.plot(node_counts, val_accs, label='Validation')
        plt.plot(node_counts, test_accs, label='Test')
        plt.xlabel("Number of Nodes")
        plt.ylabel("Accuracy")
        plt.title(f"Post-pruning curve (max_depth={max_depth})")
        plt.legend(loc='upper right')

    plt.tight_layout()

    # Ensure the output exists
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "post_prune_curves.png")
    plt.savefig(output_path)
    plt.show()
    print(f"Graphs saved as {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python c.py <train_csv> <val_csv> <test_csv> <output_folder_path>")
        sys.exit(1)
    train_csv = sys.argv[1]
    val_csv = sys.argv[2]
    test_csv = sys.argv[3]
    output_folder_path = sys.argv[4]

    run_all_curves(train_csv, val_csv, test_csv, output_folder_path)
