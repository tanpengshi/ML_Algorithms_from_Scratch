"""
Implementation of algorithm to train random forest classifiers.

Author: Tan Pengshi Alvin
Adapted from: https://towardsdatascience.com/master-machine-learning-random-forest-from-scratch-with-python-3efdd51b6d7a
"""
import numpy as np
import pandas as pd

from scipy import stats
from src.decision_tree import DecisionTree

class RandomForest:
    '''
    A class that implements Random Forest algorithm from scratch.

    For more information, refer to https://towardsdatascience.com/master-machine-learning-random-forest-from-scratch-with-python-3efdd51b6d7a

    Parameters:
    ----------    
    num_tree: int, default=5
        The number of voting decision tree classifiers used for classification.

    subsample_size: float, default=None
        The proportion of the total training examples used to train each decision trees.

    max_depth: int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until, all leaves are the purest.

    max_features: int, float, default=None
        For each decision tree, at each split from parent node to child nodes, consider only 'max features' to find threshold split. 
        If float and <1, max_features take the proportion of the features in the dataset.

    bootstrap: bool, default=True
        Bootstrap sampling of training examples, with or without replacement. 

    random_state: int, default=None
        Controls the randomness of the estimator. The features are always randomly permuted at each split in each decision tree, 
        and bootstrap sampling is randomly permuted.
    '''
    def __init__(self, num_trees=5, subsample_size=None, max_depth=None, max_features=None, bootstrap=True, random_state=None):
        self.num_trees = num_trees
        self.subsample_size = subsample_size
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        # Will store individually trained decision trees
        self.decision_trees = []

    def sample(self, X, y, random_state):
        '''
        Helper function used for boostrap sampling.
        
        Parameters:
        ----------
        X: np.array, features
        y: np.array, target
        random_state: int, random bootstrap sampling

        Returns:
        -------
        sample of features: np.array, feature bootstrapped sample
        sample of target: np.array, corresponding target bootstrapped sample
        '''
        n_rows, n_cols = X.shape

        # Sample with replacement
        if self.subsample_size is None:
            sample_size = n_rows
        else:
            sample_size = int(n_rows*self.subsample_size)

        np.random.seed(random_state)
        samples = np.random.choice(a=n_rows, size=sample_size, replace=self.bootstrap)

        return X[samples], y[samples]


    def fit(self, X, y):
        '''
        Instantiates a trained Random Forest classifier object, with the corresponding rules stored as attributes in the nodes in each
        decision tree.
        
        Parameters:
        ----------
        X: np.array or pd.core.frame.DataFrame
            The set of feature variables of the training dataset

        y: np.array or pd.core.frame.DataFrame
            The target variable of the training dataset

        Returns:
        -------
        None
        '''
        # Reset
        if len(self.decision_trees) > 0:
            self.decision_trees = []

        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
        if isinstance(y, pd.core.series.Series):
            y = y.values
            
        # Build each tree of the forest
        num_built = 0

        while num_built < self.num_trees:

            clf = DecisionTree(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=self.random_state
            )

            # Obtain data sample
            _X, _y = self.sample(X, y, self.random_state)
            # Train
            clf.fit(_X, _y)
            # Save the classifier
            self.decision_trees.append(clf)
            
            num_built += 1

            if self.random_state is not None:
                self.random_state += 1

    def predict(self, X):
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
        # Make predictions with every tree in the forest
        y = []
        for tree in self.decision_trees:
            y.append(tree.predict(X))
        
        # Reshape so we can find the most common value
        y = np.swapaxes(y, axis1=0, axis2=1)
        
        # Use majority voting for the final prediction
        predicted_classes = stats.mode(y,axis=1)[0].reshape(-1)

        return predicted_classes
