from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        cur_err = np.inf
        for i in range(X.shape[1]):
            # todo: maybe subtract y
            p_thr, p_thr_err = self._find_threshold(X[:, i], y, 1)
            if p_thr_err < cur_err:
                cur_err = p_thr_err
                self.threshold_, self.j_, self.sign_ = p_thr, i, 1

            n_thr, n_thr_err = self._find_threshold(X[:, i], y, -1)
            if n_thr_err < cur_err:
                cur_err = n_thr_err
                self.threshold_, self.j_, self.sign_ = n_thr, i, -1

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        res = (X[:, self.j_] - self.threshold_ >= 0) * self.sign_
        res[res == 0] = -self.sign_
        return res

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        feature_vals: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels_distrebuted: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # sorted_indices = np.argsort(values)
        # feature_vals, labels = values[sorted_indices], labels[sorted_indices]
        # # todo: this implementation need efficiency improvements
        # # the misclassification error
        # mse = []
        # for i in range(len(labels)):
        #     same_sign = labels[i:]
        #     other_sign = labels[:i]
        #     same_sign_dif = np.sum(np.abs(same_sign)) - sign * (np.sum(labels) - i)
        #     other_sign_dif = np.sum(other_sign) + sign * i
        #     mse.append(same_sign_dif + other_sign_dif)
        # mse = np.asarray(mse)
        # minimizer_idx = np.argmin(mse)
        #
        # return feature_vals[minimizer_idx], mse[minimizer_idx]

        # ----------------------------------------------------------------------------------------------------------
        # changing to school solution for the sake of efficiency:
        # sort the data so that x1 <= x2 <= ... <= xm
        sort_idx = np.argsort(values)
        values, labels = values[sort_idx], labels[sort_idx]

        thetas = np.concatenate([[-np.inf], (values[1:] + values[:-1]) / 2, [np.inf]])
        minimal_theta_loss = np.sum(np.abs(labels[sign != np.sign(labels)]))  # loss of the smallest possible theta

        losses = np.append(minimal_theta_loss, minimal_theta_loss + np.cumsum(labels * sign))
        # losses = np.append(minimal_theta_loss, minimal_theta_loss - np.cumsum(labels * sign))
        min_loss_idx = np.argmin(losses)
        return thetas[min_loss_idx], losses[min_loss_idx]

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
        return misclassification_error(y, self._predict(X))
