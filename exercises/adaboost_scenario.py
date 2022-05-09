import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NO_NOISE = 0


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    X_train, y_train = generate_data(train_size, noise)
    X_test, y_test = generate_data(test_size, noise)
    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(lambda: DecisionStump(), n_learners)
    adaboost.fit(X_train, y_train)

    train_err = [adaboost.partial_loss(X_train, y_train, i) for i in range(1, n_learners)]
    test_err = [adaboost.partial_loss(X_test, y_test, i) for i in range(1, n_learners)]

    fig = go.Figure([go.Scatter(x=np.arange(1, n_learners), y=train_err, mode='lines', name='Training errors'),
                     go.Scatter(x=np.arange(1, n_learners), y=test_err, mode='lines', name='Test errors')])
    fig.show()

    # Question 2: Plotting decision surfaces
    # T = [5, 50, 100, 250]
    # lims = np.array([np.r_[X_train, X_test].min(axis=0), np.r_[X_train, X_test].max(axis=0)]).T + np.array([-.1, .1])
    # raise NotImplementedError()
    #
    # # Question 3: Decision surface of best performing ensemble
    # raise NotImplementedError()
    #
    # # Question 4: Decision surface with weighted samples
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(NO_NOISE)
