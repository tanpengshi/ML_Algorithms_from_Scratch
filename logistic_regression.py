"""
Implementation of algorithm to train Logistic Regression.
Author: Tan Pengshi Alvin
"""
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Logistic_Regression:
    """
    A Logistic Regression model, as a single-layer perceptron.
    """
    def __init__(self, iterations=10000, tolerance=0.00001, learning_rate=0.00001):
        self.iter = iterations
        self.tol = tolerance
        self.lr = learning_rate
        self._w = 0
        self._b = 0

    def _sigmoid(self,z):
        """
        Parameters:
        -----------
        z: A scalar or numpy array of any size.

        Returns:
        --------
        s: sigmoid(z)
        """  
        s = 1/(1+np.exp(-z))

        return s

    def _initialize_parameters(self,n):
        """
        Parameters:
        -----------
        n: size of w vector (number of features).

        Returns:
        --------
        w -- initialized vector of shape (1, n)
        b -- initialized scalar (corresponds to the bias)
        """         
        w = np.zeros((1,n))
        b = 0

        return w, b

    def fit(self, X, Y):
        """
        Parameters:
        ----------
        X: numpy.ndarray or pandas.core.frame.DataFrame
            Training examples of input data of size (m, n)

        Y: numpy.ndarray or pandas.core.frame.DataFrame or pandas.core.series.Series
            The binary targets with the associated ground truths, of size (m,1)

        Returns:
        --------
        None        
        """
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
        if isinstance(Y, (pd.core.frame.DataFrame,pd.core.series.Series)):
            Y = Y.values

        X = X.T   # transpose values, X becomes (n,m) matrix
        Y = Y.T   # transpose values, Y becomes (1,m) matrix 

        m = X.shape[1]
        n = X.shape[0]

        w, b = self._initialize_parameters(n)

        for i in range(self.iter):
            #Forward Propagation
            A = self._sigmoid(np.dot(w,X) + b)              # compute activation
            cost = np.sum(((- np.log(A))*Y + (-np.log(1-A))*(1-Y)))/m  # compute cost
            #Backward Propagation
            dz = A - Y
            db = np.sum(dz)/m
            dw = np.dot(dz,X.T)/m
            #Update parameters
            w = w - self.lr*dw
            b = b - self.lr*db

            # Print cost every 100 training iterations
            if i % 100 == 0:
                logger.info(f"Cost after iteration {i}: {cost}")
            # check stopping criterion
            grad_vec = np.concatenate([np.squeeze(dw),[db]])
            delta = self.lr*grad_vec
            if np.linalg.norm(delta) < self.tol:
                logger.info(f"Final cost at convergence after iteration {i}: {cost}")
                break  
            elif np.linalg.norm(delta) >= self.tol and i==self.iter-1:
                logger.info(f"Final cost before convergence after iteration {i}: {cost}")
                logger.info("The max iteration was reached before convergence")
            else:
                pass

        self._w = w
        self._b = b

    
    def predict(self, X):
        """
        Parameters:
        ----------
        X: numpy.ndarray or pandas.core.frame.DataFrame
            Test examples of input data of size (m, n)

        Returns:
        -------            
        predictions: numpy.ndarray
            Predicted binary classes for each test examples
        """
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values

        X = X.T   # transpose values, X becomes (n,m) matrix

        #Forward Propagation
        A = self._sigmoid(np.dot(self._w,X) + self._b)      # compute activation

        predictions = (A >= 0.5)[0]

        return predictions