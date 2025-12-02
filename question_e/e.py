import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def load_and_preprocess(file):
    df = pd.read_csv(file)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    y = df['result'] if 'result' in df.columns else None
    X = df.drop(columns=['result']) if 'result' in df.columns else df
    categorical_cols = ['team', 'opp', 'host', 'month']
    X = pd.get_dummies(X, columns=categorical_cols)
    return X, y

def main(train_path, val_path, test_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X_train, y_train = load_and_preprocess(train_path)
    X_val, y_val = load_and_preprocess(val_path)
    X_test, _ = load_and_preprocess(test_path)

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    max_depths = [15, 25, 35, 45]
    train_accs = []
    val_accs = []

    for depth in max_depths:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        train_pred = clf.predict(X_train)
        val_pred = clf.predict(X_val)
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"Max depth {depth}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

    plt.figure()
    plt.plot(max_depths, train_accs, label='Train Accuracy')
    plt.plot(max_depths, val_accs, label='Validation Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Accuracy vs Max Depth')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'max_depth_accuracy.png'))
    plt.show()

    best_depth = max_depths[val_accs.index(max(val_accs))]
    print(f"Best max depth by validation: {best_depth}")

    ccp_alphas = [0.0, 0.0001, 0.0003, 0.0005]
    train_accs_ccp = []
    val_accs_ccp = []

    for alpha in ccp_alphas:
        clf = DecisionTreeClassifier(criterion='entropy', ccp_alpha=alpha, random_state=42)
        clf.fit(X_train, y_train)
        train_pred = clf.predict(X_train)
        val_pred = clf.predict(X_val)
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        train_accs_ccp.append(train_acc)
        val_accs_ccp.append(val_acc)
        print(f"ccp_alpha {alpha}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

    plt.figure()
    plt.plot(ccp_alphas, train_accs_ccp, label='Train Accuracy')
    plt.plot(ccp_alphas, val_accs_ccp, label='Validation Accuracy')
    plt.xlabel('ccp_alpha')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Accuracy vs CCP Alpha')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'ccp_alpha_accuracy.png'))
    plt.show()

    best_alpha = ccp_alphas[val_accs_ccp.index(max(val_accs_ccp))]
    print(f"Best ccp_alpha by validation: {best_alpha}")

    final_clf_depth = DecisionTreeClassifier(criterion='entropy', max_depth=best_depth, random_state=42)
    final_clf_depth.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    test_pred_depth = final_clf_depth.predict(X_test)

    final_clf_ccp = DecisionTreeClassifier(criterion='entropy', max_depth=None, ccp_alpha=best_alpha, random_state=42)
    final_clf_ccp.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    test_pred_ccp = final_clf_ccp.predict(X_test)

    pd.DataFrame({'result': test_pred_depth}).to_csv(os.path.join(output_dir, 'test_pred_best_depth.csv'), index=False)
    pd.DataFrame({'result': test_pred_ccp}).to_csv(os.path.join(output_dir, 'test_pred_best_ccp_alpha.csv'), index=False)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python e.py <train.csv> <val.csv> <test.csv> <output_folder>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
