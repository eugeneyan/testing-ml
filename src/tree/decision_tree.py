"""
DecisionTree class
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.utils.logger import logger


class Node:
    """Class for decision tree node."""

    def __init__(self, left: Node, right: Node, split_function: Callable, leaf_label: int = None):
        """Initializes a node that splits data into left and right nodes based on split_function.

        Args:
            left: Left child node
            right: Right child node
            split_function: Callable to decide left or right node
            leaf_label: Label for leaf node
        """
        self.left = left
        self.right = right
        self.split_function = split_function
        self.leaf_label = leaf_label

    def __str__(self):  # pragma: no cover
        return self.__class__.__name__

    def split(self, feature: np.array) -> Node:
        """Returns a child node based on split function

        Args:
            feature: Feature vector

        Returns:
            If leaf node, returns a leaf label. Else, a child node.
        """
        if self.leaf_label is not None:
            return self.leaf_label  # type: ignore

        else:
            if self.split_function(feature):
                return self.left.split(feature)
            else:
                return self.right.split(feature)


def gini_impurity(labels: np.array) -> float:
    """Returns the gini impurity for a lits of labels

    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity

    Args:
        labels: Array of binary labels (1 or 0)

    Returns:
        Gini impurity
    """
    label_array = np.array(labels)

    prob_pos = len(np.where(label_array == 1)[0]) / len(label_array)
    prob_neg = len(np.where(label_array == 0)[0]) / len(label_array)

    return 1 - (prob_pos ** 2 + prob_neg ** 2)


def gini_gain(prev_labels: np.array, labels: np.array) -> float:
    """Returns the information gain between the previous and current set of labels

    https://en.wikipedia.org/wiki/Decision_tree_learning#Information_gain

    Args:
        prev_labels: List of binary labels (1 or 0)
        labels: List of List of binary labels (1 or 0)

    Returns:
        Information gain
    """
    gini_prev = gini_impurity(prev_labels)
    len_prev = len(prev_labels)

    gini_current = [gini_impurity(vec) for vec in labels if len(vec) > 0]
    weights = [len(vec) / len_prev for vec in labels if len(vec) > 0]

    return gini_prev - sum([weight * gini for weight, gini in zip(weights, gini_current)])


class DecisionTree:
    """Class for decision tree."""

    def __init__(self, depth_limit: int = 99):
        """Initializes a decision tree with depth limit

        Args:
            depth_limit: Maximum depth to build the tree
        """
        self.root = None
        self.depth_limit = depth_limit
        logger.info('{} initialized with depth limit: {}'.format(self.__class__.__name__, depth_limit))

    def check_stopping_condition(self, labels: np.array, depth: int) -> bool:
        """Checks if stopping conditions (i.e., labels are all same, depth limit reached) has been met

        Args:
            labels: Vector of labels
            depth: Current depth of tree

        Returns:
            True if stopping condition has been met, False otherwise.
        """
        if depth == self.depth_limit:  # Depth has been reached
            return True
        elif len(set(labels)) == 1:  # All labels are the same
            return True
        else:
            return False

    @staticmethod
    def get_percentile_list(n_percentiles: int = 100) -> np.array:
        """Returns an array of percentiles. Note: This is used in np.percentile, which requires the sequence of
        percentiles to be between 0 to 100 inclusive.

        Args:
            n_percentiles: Number of percentiles to split on

        Returns:
            Array of percentiles
        """
        return np.arange(1, 100, 100 / n_percentiles)[1:]

    @staticmethod
    def get_probability(labels: np.array) -> float:
        """Returns the majority label

        Args:
            labels:

        Returns:

        """
        if len(labels) > 0:
            return labels.mean()
        return None  # type: ignore  # pragma: no cover

    def __build_tree__(self, features: np.array, labels: np.array, depth: int = 0) -> Node:
        """Build decision tree that learns split functions.

        Args:
            features:
            labels:
            depth:

        Returns:
            Decision tree root node
        """
        n_percentiles = max(2, int(len(labels) / 10))

        if self.check_stopping_condition(labels, depth):
            prob = self.get_probability(labels)
            return Node(None, None, None, prob)  # type: ignore

        else:
            logger.debug('Features: {}'.format(features))
            logger.debug('Labels: {}'.format(labels))
            splits = np.percentile(features, self.get_percentile_list(n_percentiles), axis=0)

            best_split = None
            best_split_feat_idx = None
            best_split_gini_gain = float('-inf')

            for feat_idx, feat_col in enumerate(features.T):  # Transpose to loop through columns
                logger.debug('Col index: {}'.format(feat_idx))

                for split in splits[:, feat_idx]:
                    labels_left = labels[np.where(feat_col < split)]
                    labels_right = labels[np.where(feat_col >= split)]

                    gain = gini_gain(labels, [labels_left, labels_right])

                    if gain > best_split_gini_gain:
                        best_split_gini_gain, best_split, best_split_feat_idx = gain, split, feat_idx

            split_left = np.where(features[:, best_split_feat_idx] < best_split)
            split_right = np.where(features[:, best_split_feat_idx] >= best_split)
            logger.debug('Split left: {} | right: {}'.format(split_left, split_right))

            features_left, features_right = features[split_left], features[split_right]
            labels_left, labels_right = labels[split_left], labels[split_right]

            # If either node is empty after splitting
            if len(labels_left) == 0:
                node_left = Node(None, None, None, self.get_probability(labels))  # type: ignore
                node_right = self.__build_tree__(features_right, labels_right, depth + 1)
            elif len(labels_right) == 0:  # pragma: no cover
                node_left = self.__build_tree__(features_left, labels_left, depth + 1)
                node_right = Node(None, None, None, self.get_probability(labels))  # type: ignore
            else:
                node_left = self.__build_tree__(features_left, labels_left, depth + 1)
                node_right = self.__build_tree__(features_right, labels_right, depth + 1)

        return Node(node_left, node_right, lambda feature: feature[best_split_feat_idx] < best_split)

    def fit(self, features: np.array, labels: np.array):
        """Build decision tree

        Args:
            features:
            labels:

        Returns:
            None
        """
        self.root = self.__build_tree__(features, labels)  # type: ignore

    def predict(self, features: np.array) -> np.array:
        """Returns labels given a set of features.

        Args:
            features: Numpy array of features with shape (row x col)

        Returns:
            Predicted labels
        """
        labels = np.array([self.root.split(row) for row in features])  # type: ignore

        return labels
