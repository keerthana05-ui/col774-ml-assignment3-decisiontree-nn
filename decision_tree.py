import numpy as np
import pandas as pd
from collections import Counter

class TreeNode:
    def __init__(self, depth=0, max_depth=None):
        self.is_leaf = False
        self.prediction = None
        self.split_attr = None
        self.split_val = None
        self.children = dict()
        self.depth = depth
        self.max_depth = max_depth

def entropy(y):
    counts = Counter(y)
    total = len(y)
    return -sum((c / total) * np.log2(c / total) for c in counts.values() if c > 0)

def gini(y):
    counts = Counter(y)
    total = len(y)
    return 1 - sum((c / total)**2 for c in counts.values() if c > 0)

def information_gain(X, y, attr, split_val=None, criterion="entropy"):
    if split_val is None:
        uniques = X[attr].unique()
        total_metric = entropy(y) if criterion == "entropy" else gini(y)
        weighted_metric = 0
        for v in uniques:
            mask = (X[attr] == v)
            subset_y = y[mask]
            if len(subset_y) > 0:
                metric = entropy(subset_y) if criterion == "entropy" else gini(subset_y)
                weighted_metric += (len(subset_y) / len(y)) * metric
        return total_metric - weighted_metric
    else:
        left_mask = X[attr] <= split_val
        right_mask = X[attr] > split_val
        left_y = y[left_mask]
        right_y = y[right_mask]
        total_metric = entropy(y) if criterion == "entropy" else gini(y)
        left_metric = entropy(left_y) if criterion == "entropy" else gini(left_y)
        right_metric = entropy(right_y) if criterion == "entropy" else gini(right_y)
        weighted_metric = (len(left_y) / len(y)) * left_metric + (len(right_y) / len(y)) * right_metric
        return total_metric - weighted_metric

def best_split(X, y, criterion="entropy"):
    best_gain = -np.inf
    best_attr = None
    best_val = None
    for attr in X.columns:
        if attr == "Unnamed":
            continue
        if X[attr].dtype == 'object':
            gain = information_gain(X, y, attr, criterion=criterion)
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
                best_val = None
        else:
            median_val = X[attr].median()
            gain = information_gain(X, y, attr, split_val=median_val, criterion=criterion)
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
                best_val = median_val
    return best_attr, best_val

class DecisionTree:
    def __init__(self, max_depth=None, criterion="entropy"):
        self.max_depth = max_depth
        self.root = None
        self.criterion = criterion

    def fit(self, X, y):
        majority_class = Counter(y).most_common(1)[0][0]
        self.root = self._build_tree(X, y, depth=0, parent_majority=majority_class, criterion=self.criterion)

    def _build_tree(self, X, y, depth, parent_majority, criterion="entropy"):
        node = TreeNode(depth, self.max_depth)
        if len(y) == 0:
            node.is_leaf = True
            node.prediction = parent_majority
            return node
        if len(set(y)) == 1 or len(X) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            node.is_leaf = True
            node.prediction = Counter(y).most_common(1)[0][0]
            return node

        split_attr, split_val = best_split(X, y, criterion=criterion)
        if split_attr is None:
            node.is_leaf = True
            node.prediction = Counter(y).most_common(1)[0][0]
            return node

        node.split_attr = split_attr
        node.split_val = split_val

        new_parent_majority = Counter(y).most_common(1)[0][0]

        if split_val is None:
            for value in X[split_attr].unique():
                child_mask = X[split_attr] == value
                node.children[value] = self._build_tree(
                    X[child_mask].drop(split_attr, axis=1),
                    y[child_mask],
                    depth + 1,
                    new_parent_majority,
                    criterion=criterion
                )
        else:
            left_mask = X[split_attr] <= split_val
            right_mask = X[split_attr] > split_val
            node.children['leq'] = self._build_tree(X[left_mask], y[left_mask], depth + 1, new_parent_majority, criterion)
            node.children['gt'] = self._build_tree(X[right_mask], y[right_mask], depth + 1, new_parent_majority, criterion)
        return node

    def predict_one(self, x, node=None):
        if node is None:
            node = self.root
        if node.is_leaf:
            return node.prediction
        attr = node.split_attr
        if node.split_val is None:
            val = x.get(attr)
            if val in node.children:
                return self.predict_one(x, node.children[val])
            else:
                return Counter(child.prediction for child in node.children.values()).most_common(1)[0][0]
        else:
            if x[attr] <= node.split_val:
                return self.predict_one(x, node.children['leq'])
            else:
                return self.predict_one(x, node.children['gt'])

    def predict(self, X):
        if 'Unnamed' in X.columns:
            X = X.drop(columns=['Unnamed'])
        return X.apply(lambda row: self.predict_one(row), axis=1)

def accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))
