from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    split_X = np.array_split(X, cv)
    split_y = np.array_split(y, cv)
    Average_train_scores, Average_validation_scores = [], []
    for i in range(cv):
        folds_train_X = np.concatenate(np.delete(split_X, i, axis=0))
        folds_train_y = np.concatenate(np.delete(split_y, i, axis=0))

        estimator.fit(folds_train_X, folds_train_y)

        Average_train_scores.append(scoring(folds_train_y, estimator.predict(folds_train_X)))
        Average_validation_scores.append(scoring(split_y[i], estimator.predict(split_X[i])))

    return np.mean(Average_train_scores), np.mean(Average_validation_scores)
