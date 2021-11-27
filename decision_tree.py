"""
Implementation of algorithm to train decision tree classifiers.

Author: Tan Pengshi Alvin
Adapted from: https://towardsdatascience.com/decision-tree-from-scratch-in-python-46e99dfea775
"""
import numpy as np
import pandas as pd
import random

class Node:
    """
    Node object of the decision tree. Each node may contain other node objects as attributes, as the decision tree grows.
    An exception is when the decision tree has reached the terminal node.

    Parameters:
    ----------
    predicted_class: int
        the predicted class is specified by taking the mode of the classes in the node during training. Predicted class is an 
        important information to capture in the terminal node.
    
    Attributes:
    ----------
    feature_index: int
        The index of the feature of the fitted data where the split will occur for the node

    threshold: float
        The value split ('less than' and 'more than') for the chosen feature

    left: <class Node>
        the left child Node that will be grown that fufills the condition 'less than' threshold

    right: <class Node>
        the right child Node that will be grown that fulfils the condition 'more than' threshold
    """
    def __init__(self,predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        

class DecisionTree:
    """
    A decision tree classifier. 

    For more information, refer to https://towardsdatascience.com/decision-tree-from-scratch-in-python-46e99dfea775

    Parameters
    ----------
    max_depth: int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until, all leaves are the purest.

    max_features: int, float, default=None
        At each split from parent node to child nodes, consider only 'max features' to find threshold split. If float and <1, 
        max_features take the proportion of the features in the dataset.

    random_state: int, default=None
        Controls the randomness of the estimator. The features are always randomly permuted at each split. 
        When ``max_features < n_features``, the algorithm will select ``max_features`` at random at each split,
        before finding the best split among them. But the best found split may vary across different runs, 
        even if ``max_features=n_features``. That is the case, if the improvement of the criterion is identical for several splits
        and one split has to be selected at random.

    Attributes:
    ----------
    tree: <class Node>
        The root node which obtains all other sub-nodes, which are recursively stored as attributes.
    """
    def __init__(self,max_depth=None,max_features=None,random_state=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.tree = None

    def fit(self, X, y):
        """
        Instantiates a trained Decision Tree Classifier object, with the corresponding rules stored as attributes in the nodes.

        Parameters:
        ----------     
        X: np.array or pd.core.frame.DataFrame
            The set of feature variables of the training dataset
        
        y: np.array or pd.core.series.Series
            The target variable of the training dataset

        Returns:
        -------
        None
        """
        # store number of classes and features of the dataset into model object
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
        if isinstance(y, pd.core.series.Series):
            y = y.values

        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        if self.max_features==None:
            self.max_features = self.n_features

        if isinstance(self.max_features,float) and self.max_features<=1:
            self.max_features = int(self.max_features*self.n_features)

        # create tree for the dataset
        self.tree = self.grow_tree(X,y,self.random_state)


    def predict(self,X):
        """
        Predict class for each test example in a test set.

        Parameters:
        ----------     
        X: np.array or pd.core.frame.DataFrame
            The set of feature variables of the test dataset      

        Returns:
        -------
        predicted_classes: np.array
            The numpy array of predict class for each test example      
        """
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values

        predicted_classes = np.array([self.predict_example(inputs) for inputs in X])

        return predicted_classes


    def best_split(self, X, y, random_state):
        """
        Obtains the optimal feature index and threshold value for the split at the parent node, which are then used to decide the split of
        training examples of features/targets into smaller subsets.

        Parameters:
        ----------     
        X: np.array
            Subset of all the training examples of features at the parent node.

        y: np.array
            Subset of all the training examples of targets at the parent node.

        random_state: int, default=None

        Returns:
        -------
        best_feat_id: int, None
            The feature index considered for split at parent node.
        
        best_threshold: float, None
            The threshold value at the feature considered for split at parent node.
        """
        m = len(y)
        if m <= 1:
            return None, None

        num_class_parent = [np.sum(y==c) for c in range(self.n_classes)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_class_parent)
        if best_gini == 0:
            return None, None

        best_feat_id, best_threshold = None, None

        random.seed(random_state)
        feat_indices = random.sample(range(self.n_features),self.max_features)

        for feat_id in feat_indices:

            sorted_column = sorted(set(X[:,feat_id]))
            threshold_values = [np.mean([a,b]) for a,b in zip(sorted_column,sorted_column[1:])]

            for threshold in threshold_values:

                left_y = y[X[:,feat_id]<threshold]
                right_y = y[X[:,feat_id]>threshold]

                num_class_left = [np.sum(left_y==c) for c in range(self.n_classes)]
                num_class_right = [np.sum(right_y==c) for c in range(self.n_classes)]

                gini_left = 1.0 - sum((n / len(left_y)) ** 2 for n in num_class_left)
                gini_right = 1.0 - sum((n / len(right_y)) ** 2 for n in num_class_right)

                gini = (len(left_y)/m)*gini_left + (len(right_y)/m)*gini_right

                if gini < best_gini:
                    best_gini = gini
                    best_feat_id = feat_id
                    best_threshold = threshold

        return best_feat_id, best_threshold


    def grow_tree(self, X, y, random_state, depth=0):
        """
        Recursive function to continuously generate nodes. At each recursion step, a parent node is formed and recursively split 
        into left child node and right child node IF the maximum depth is not reached or the parent node is less pure than
        the average gini of child nodes.

        Parameters:
        ----------     
        X: np.array
            Subset of all the training examples of features at the parent node.

        y: np.array
            Subset of all the training examples of targets at the parent node.

        random_state: int, default=None

        depth: int
            The number of times a branch has split.

        Returns:
        --------
        node: <class Node>
            The instantiated Node, with its corresponding attributes.
        """
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)

        node = Node(predicted_class=predicted_class)

        if (self.max_depth is None) or (depth < self.max_depth):
            id, thr = self.best_split(X, y, random_state)

            if id is not None:
                if random_state is not None:
                    random_state += 1

                indices_left = X[:, id] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                
                node.feature_index = id
                node.threshold = thr
                node.left = self.grow_tree(X_left, y_left, random_state, depth + 1)
                node.right = self.grow_tree(X_right, y_right, random_state, depth + 1)

        return node      


    def predict_example(self, inputs):
        """
        Generate the predicted class of a single row of test exmaple based on the feature indices and thresholds that have been stored
        in all the nodes.

        Parameters:
        ----------     
        inputs: An row of test examples containing the all the features that have been trained on.

        Returns:
        --------
        node.predicted_class: int
            The stored attribute - predicted_class - of the terminal node.
        """
        node = self.tree

        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right

        return node.predicted_class     

        
        

    