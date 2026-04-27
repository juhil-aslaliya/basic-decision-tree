import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    def __init__(self, min_samples_split=2, max_features=None, max_depth=100, features=None, result_name = None, result_vals = None):
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.max_depth = max_depth
        self.root = None
        self.features = features
        self.result_name = result_name
        self.result_vals = result_vals

    def _entropy(self, y):
        counts = np.bincount(y)
        probs = counts/len(y)
        entropy = -np.sum([p*np.log2(p) for p in probs if p > 0])
        return entropy

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _information_gain(self, y, X_column, split_thresh):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r
        ig = parent_entropy - child_entropy
        return ig

    def _best_split(self, X, y):
        best_split = {'gain': 0.0, 'feature': None, 'threshold': None}
        n_samples, n_features = X.shape
        if self.max_features is not None:
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        else:
            feature_indices = range(n_features)
        for feat_idx in feature_indices:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_split['gain']:
                    best_split['gain'] = gain
                    best_split['feature'] = feat_idx
                    best_split['threshold'] = threshold
        return best_split['feature'], best_split['threshold']

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        if n_samples == 0:
            return Node(value=0)
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)
        best_feat, best_thresh = self._best_split(X, y)
        if best_feat is None:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(feature_index=best_feat, threshold=best_thresh, left=left, right=right)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def print_tree(self, node=None, prefix='', is_left=True, is_root=True):
        if node is None:
            node = self.root
            if node is None:
                print("The tree has not been trained yet!")
                return
        if is_root:
            connector=''
            branch_label=''
        else:
            connector = "├── " if is_left else "└── "
            branch_label = 'True: ' if is_left else 'False: '
        if node.value is not None:
            result_name = self.result_name if self.result_name is not None else 'Class'
            node_val = self.result_vals[node.value] if self.result_vals is not None else node.value
            print(prefix + connector + branch_label + f'Predict: {result_name} = {node_val}')
            return
        feat_name = self.features[node.feature_index] if self.features else f'Feature {node.feature_index}'
        print(prefix + connector + branch_label + f'[{feat_name} <= {node.threshold:.4f}]')
        if is_root:
            new_prefix = ''
        else:
            new_prefix = prefix + ("│   " if is_left else "    ")
        self.print_tree(node.left, new_prefix, is_left=True, is_root=False)
        self.print_tree(node.right, new_prefix, is_left=False, is_root=False)


class RandomForest:
    def __init__(self, n_trees=10, max_depth=3, min_samples_split=2, max_features=4, features=None, result_name=None, result_vals=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.features = features
        self.result_name = result_name
        self.result_vals = result_vals
        self.trees = []

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.max_features, features=self.features, result_name=self.result_name, result_vals=self.result_vals)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def _majority_vote(self, predictions):
        counter = Counter(predictions)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        return np.array([self._majority_vote(preds) for preds in tree_preds])