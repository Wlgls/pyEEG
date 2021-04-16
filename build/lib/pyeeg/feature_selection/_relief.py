# -*- encoding: utf-8 -*-
'''
@File        :__init__.py
@Time        :2021/03/30 17:00:33
@Author      :wlgls
@Version     :1.0
'''

from sklearn.neighbors import KDTree
import numpy as np

class Relief(object):
    """This is the code implementation of relief, It is modified from https://github.com/gitter-badger/ReliefF/.

    Relief algorithm is a kind of feature weighting algorithms, which gives different weights to each feature according to the correlation of each feature and category. Then the feature is selected according to the weight

    Parameters
    ----------
    m_samples int: , optional
        In the relief algorithm, we need to sample from the training samples many times, and it's the number of samples. If it's None, It's X.shape[0]. by default None
    n_features_to_keep : int, optional
        The number of features you keep. If it's None, It's X.shape[2]//2.  by default None
    feature_scores: 1d-array
        The weight of the feature.
    feature_sort: 1d-array
        Index of features descending sorted by score, 

    Examples
    ---------
    """
    def __init__(self, m_samples=None, n_features_to_keep=None):
        
        self.m_samples = m_samples
        self.n_features_to_keep =n_features_to_keep
        self.feature_scores = None
        self.feature_sort = None

    def fit(self, X, y):
        """Learn the features to select.

        Parameters
        ----------
        X : array-like
            Training vectors
        y : array-like
            Target values

        Return
        ----------
        self: object
        """

        if self.m_samples is None:
            self.m_samples = X.shape[0]
        if self.n_features_to_keep is None:
            self.n_features_to_keep = X.shape[1] // 2

        self.feature_scores = np.zeros(X.shape[1])
        
        # binary classification problem
        l1, l2 = np.unique(y)[0], np.unique(y)[1]
        l1X = X[y==l1]
        l2X = X[y==l2]
        l1Tree = KDTree(l1X)
        l2Tree = KDTree(l2X)

        for source_ind in np.random.choice(X.shape[0], size=self.m_samples, replace=False):
            
            # In order to eliminate the interference of the same elements, k = 2 is selected
            _, l1ind = l1Tree.query(X[source_ind].reshape(1, -1), k=2)
            _, l2ind = l2Tree.query(X[source_ind].reshape(1, -1), k=2)

            # del the same elements
            if np.any(X[source_ind] == l1X[l1ind[0, 0]]):
                l1ind = l1ind[0, 1]
                l2ind = l2ind[0, 0]
            else:
                l1ind = l1ind[0, 0]
                l2ind = l2ind[0, 1]

            # Weight update
            if y[source_ind] == l1:
                # It means l1ind is hit
                self.feature_scores = self.feature_scores - l1X[l1ind]/self.m_samples + l2X[l2ind]/self.m_samples
            if y[source_ind] == l2:
                # It means l2ind is hit
                self.feature_scores = self.feature_scores + l1X[l1ind]/self.m_samples - l2X[l2ind]/self.m_samples
            
        self.feature_sort = np.argsort(self.feature_scores)[::-1]

        return self

    def transform(self, X):
        """select features

        Parameters
        ----------
        X : array-like
            Test vectors

        Return
        ----------
        X_reduced: array-like
            Selected features
        """
        return X[:, self.feature_sort[:self.n_features_to_keep]]

            

            
