from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        self.classes_, counts = np.unique(y, return_counts=True)
        pi_k = dict(zip(self.classes_, counts))

        self.pi_ = counts / len(y)
        assert np.sum(self.pi_) > 0.99
        assert np.sum(self.pi_) < 1.01

        self.mu_ = np.asarray([np.sum(X[np.where(y == k)], axis=0) / pi_k[i] for i, k in enumerate(self.classes_)])

        inner_prod = []
        # enumerate for cases which k !=  i indices
        for i, k in enumerate(self.classes_):
            # summing according to k (the label index) instead of  the sample index)
            v = (X[y == k] - self.mu_[i]).T
            inner_prod.append(v @ v.T)

        # self.cov_ = np.sum(np.asarray(inner_prod), axis=0) / (X.shape[0] )
        self.cov_ = np.asarray(np.sum(np.asarray(inner_prod), axis=0) / (X.shape[0] - len(self.classes_)))
        self._cov_inv = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return np.asarray([self.classes_[k] for k in np.argmax(self.likelihood(X), axis=1)])

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        result = []
        for k in self.classes_:
            a = X @ self._cov_inv @ self.mu_[k]
            b = -0.5 * self.mu_[k] @ self._cov_inv @ self.mu_[k]
            result.append(np.log(self.pi_[k]) + a + b)

        return np.asarray(result).T

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """

        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
