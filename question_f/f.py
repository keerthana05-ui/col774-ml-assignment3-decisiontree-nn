import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

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

    param_grid = {
        'n_estimators': [50, 150, 250, 350],
        'max_features': [0.1, 0.3, 0.5, 0.7, 0.9],
        'min_samples_split': [2, 4, 6, 8, 10]
    }

    rf = RandomForestClassifier(criterion='entropy', oob_score=True, random_state=42, n_jobs=-1)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    print('Starting grid search...')
    grid.fit(X_train, y_train)
    print('Grid search done.')

    print(f"\nBest Params: {grid.best_params_}\n")

    best_rf = grid.best_estimator_
    train_pred = best_rf.predict(X_train)
    val_pred = best_rf.predict(X_val)
    test_pred = best_rf.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    oob_acc = best_rf.oob_score_ if hasattr(best_rf, "oob_score_") else None

    print(f"Training accuracy:   {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    if oob_acc is not None:
        print(f"OOB accuracy:        {oob_acc:.4f}")
    else:
        print("OOB accuracy not available.")

    # Save test predictions
    pd.DataFrame({'result': test_pred}).to_csv(os.path.join(output_dir, 'rf_test_pred.csv'), index=False)

    # Save results
    with open(os.path.join(output_dir, 'rf_results.txt'), 'w') as f:
        f.write(f"Best Params: {grid.best_params_}\n")
        f.write(f"Train Acc: {train_acc:.4f}\n")
        f.write(f"Validation Acc: {val_acc:.4f}\n")
        f.write(f"OOB Acc: {oob_acc:.4f}\n" if oob_acc is not None else "OOB Acc: N/A\n")

    # Plot (feature importances)
    feat_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns)
    feat_importances.nlargest(20).plot(kind='barh')
    plt.title("Top 20 Feature Importances (RF)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rf_feature_importance.png'))
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python f.py <train.csv> <val.csv> <test.csv> <output_folder>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
