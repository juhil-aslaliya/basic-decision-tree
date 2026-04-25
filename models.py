from platform import node
import numpy as np
# from make_data import X, y

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

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
        best_split = {'gain': -1, 'feature': None, 'threshold': None}
        n_samples, n_features = X.shape
        for feat_idx in range(n_features):
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
    
    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root
            if node is None:
                print("The tree has not been trained yet!")
        if node.value is not None:
            print('  ' * depth + f'Predict: Class {node.value}')
            return
        print('  ' * depth + f'[Feature {node.feature_index} <= {node.threshold:.4f}]')
        print('  ' * (depth + 1) + '--> True:')
        self.print_tree(node.left, depth + 2)
        print('  ' * (depth + 1) + '--> False:')
        self.print_tree(node.right, depth + 2)


# if __name__ == '__main__':
#     tree = DecisionTree()
#     initial_chaos = tree._entropy(y)
#     print(f'number of samples = {len(y)}')
#     print(f'distribution = {np.bincount(y)}')
#     print(f'initial entropy = {initial_chaos:.4f}')