import numpy as np
from numpy.random import seed


class AdalineSGD(object):
    """ADAptive LInear NEuron stochastic classifier"""

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.cost = []
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        """
        Fit training data
        Parameters
        ----------
            X : {array-like}, shape = [n_samples, n_features}
            y : array-like, shape = [n_samples]
        Returns
        ----------
            self: object
        """

        self._initialize_weights(X.shape[1])
        self.cost = []

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
                _cost = []
                for xi, target in zip(X, y):
                    _cost.append(self._update_weights(xi, target))
                    _avg_cost = sum(_cost) / len(y)
                    self.cost.append(_avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for x1, target in zip(X,y):
                self._update_weights(x1, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.w = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        _output = self.net_input(xi)
        _error = (target - _output)
        self.w[1:] += self.eta * xi.dot(_error)
        self.w[0] += self.eta * _error
        _cost = 0.5 * _error**2
        return _cost

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w[1:]) + self.w[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
