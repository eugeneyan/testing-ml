"""
RandomForest class
"""
from typing import List

import numpy as np

from src.tree.decision_tree import DecisionTree, Node
from src.utils.logger import logger


class RandomForest(DecisionTree):
    """Class for RandomForest (of DecisionTrees)."""

    def __init__(self, num_trees: int, row_subsampling: float, col_subsampling: float,
                 depth_limit: int = 99, seed: int = 1368):
        """Initializes a decision tree with depth limit

        Args:
            num_trees:
            row_subsampling:
            col_subsampling:
            depth_limit: Maximum depth to build the tree
        """
        super().__init__(depth_limit)
        self.trees: List[Node] = []
        self.num_trees = num_trees
        self.row_subsampling = row_subsampling
        self.col_subsampling = col_subsampling
        self.col_idxs: List[np.array] = []
        np.random.seed(seed)

    def fit(self, features: np.array, labels: np.array) -> None:
        """Builds a random forest of decision trees.

        Args:
            features:
            labels:

        Returns:
            None
        """
        n_rows, n_cols = features.shape

        for i in range(self.num_trees):
            logger.debug('{} training tree: {}'.format(self.__class__.__name__, i + 1))
            shuffled_row_idx = np.random.permutation(n_rows)
            shuffled_col_idx = np.random.permutation(n_cols)

            row_idx = np.random.choice(shuffled_row_idx, int(self.row_subsampling * n_rows), replace=False)
            col_idx = np.random.choice(shuffled_col_idx, int(self.col_subsampling * n_cols), replace=False)
            self.col_idxs.append(col_idx)

            features_subsampled = features[np.ix_(row_idx, col_idx)]
            labels_subsampled = labels[row_idx]

            self.trees.append(self.__build_tree__(features_subsampled, labels_subsampled))

    def predict(self, features: np.array) -> np.array:
        """Returns labels given a set of features.

        Args:
            features: Numpy array of features with shape (row x col)

        Returns:
            Predicted labels
        """
        labels_list = []

        for tree, col_idx in zip(self.trees, self.col_idxs):
            logger.debug('Col index: {}'.format(col_idx))
            features_subsampled = features[:, col_idx]
            labels_list.append([tree.split(row) for row in features_subsampled])

        labels = np.array(labels_list).mean(axis=0)

        return labels
