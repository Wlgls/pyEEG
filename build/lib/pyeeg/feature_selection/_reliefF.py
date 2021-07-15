# -*- encoding: utf-8 -*-
'''
@File        :_reliefF.py
@Time        :2021/04/29 20:12:15
@Author      :wlgls
@Version     :1.0
'''

import numpy as np
from sklearn.neighbors import KDTree


class ReliefF(object):
    # ReliefF
    def __init__(self, m_samples=None, n_neighbors=20, n_features_to_keep=20):
        self.feature_score = None
        self.top_features = None
        self.n_neighbors = n_neighbors
        self.n_features_to_keep = n_features_to_keep


    def fit(self, X, y):
        self.feature_score = np.zeros(X.shape[1])
        max_min = np.max(X, axis=0) - np.min(X, axis=0)
        for source_ind in np.random.choice(X.shape[0], size=self.m_samples, replace=False):
            Pr = np.sum(y==y[source_ind])/len(y)
            for label in np.unique(y):
                lTree = KDTree(X[y==label])
                _, lind = lTree.query(X[source_ind].reshape(1, -1), k=self.n_neighbors+1)

                if label == y[source_ind]:
                    #If it's the same label, you need to remove the first identical element
                    lind = lind[0, 1:]
                    # compute diff
                    diff = np.abs(X[lind] - X[source_ind])/max_min
                    # update weights
                    self.feature_score += np.sum(diff, axis=0)/(self.m_samples*self.n_neighbors)
                else:
                    lind = lind[0, :-1]
                    diff = np.abs(X[lind] - X[source_ind])/max_min
                    Pc = np.sum(y==label) / len(y)

                    self.feature_score -= Pc/(1-Pr)*np.sum(diff, axis=0)/(self.m_samples*self.n_neighbors)

        self.top_features = np.argsort(self.feature_score)[::-1]

        return self

    def transform(self, X):
        return X[:, self.top_features[:self.n_features_to_keep]]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


if __name__ == "__main__":
    X = np.arange(100).reshape(10, 10)
    y = np.ones(10)
    y[[1, 3, 4, 5]] = 0
    reff = ReliefF(m_samples=2, n_features_to_keep=2, n_neighbors=2)

    reff.fit(X, y)
    print(reff.transform(X))
                    






