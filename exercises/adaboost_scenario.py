import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NO_NOISE = 0
NOISE = 0.4


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
    from IMLearn.metrics import accuracy
    (train_X, train_Y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    adaboost = AdaBoost(lambda: DecisionStump(), n_learners)
    adaboost.fit(train_X, train_Y)

    train_err = [adaboost.partial_loss(train_X, train_Y, i) for i in range(1, n_learners)]
    test_err = [adaboost.partial_loss(test_X, test_y, i) for i in range(1, n_learners)]
    # Question 1: Train- and test errors of AdaBoost in noiseless case
    fig = go.Figure([go.Scatter(x=np.arange(1, n_learners), y=train_err, mode='lines', name='Train error'),
                     go.Scatter(x=np.arange(1, n_learners), y=test_err, mode='lines', name='Test error')])
    fig.update_layout(title=f"AdaBoost's loss as a function of learners number.\n noise={noise}",
                      xaxis=dict(title="Learners number"),
                      yaxis=dict(title="loss"))
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{i} learners" for i in T])

    m = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False, name="Label 1",
                   marker=dict(color=(test_y == 1).astype(int), symbol=class_symbols[test_y.astype(int)],
                               colorscale=[custom[0], custom[-1]], line=dict(color="black", width=1)))
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(lambda x: adaboost.partial_predict(x, t), lims[0], lims[1], showscale=False),
                        m], rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(width=800, height=900,
                      title=rf"Decision Boundaries as a function of learners number: noise={noise}",
                      margin=dict(t=100))
    fig.update_xaxes(matches='x', range=[-1, 1], constrain="domain")
    fig.update_yaxes(matches='y', range=[-1, 1], constrain="domain", scaleanchor="x", scaleratio=1)

    fig.show()

    # Question 3: Decision surface of best performing ensemble
    lowest_err_size = np.argmin(test_err) + 1
    lowest_err_acc = np.round(accuracy(test_y, adaboost.partial_predict(test_X, lowest_err_size + 1)), 3)

    fig = go.Figure([decision_surface(lambda x: adaboost.partial_predict(x, lowest_err_size), lims[0], lims[1],
                                      showscale=False), m])
    fig.update_xaxes(matches='x', range=[-1, 1], constrain="domain")
    fig.update_yaxes(matches='y', range=[-1, 1], constrain="domain", scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title_text=f"Ensemble with size= {lowest_err_size} achieved the lowest test error= {lowest_err_acc}")
    fig.show()

    # # Question 4: Decision surface with weighted samples
    fig = go.Figure([decision_surface(adaboost.predict, lims[0], lims[1], showscale=False),
                     go.Scatter(x=train_X[:, 0],
                                y=train_X[:, 1],
                                mode="markers",
                                showlegend=False,
                                marker=dict(color=(train_Y == 1).astype(int),
                                            size=adaboost.D_ / np.max(adaboost.D_) * 5,
                                            symbol=class_symbols[train_Y.astype(int)],
                                            colorscale=[custom[0], custom[-1]],
                                            line=dict(color="black", width=1)))])

    fig.update_xaxes(range=[-1, 1], constrain="domain")
    fig.update_yaxes(range=[-1, 1], constrain="domain", scaleanchor="x", scaleratio=1)
    fig.update_layout(dict1=dict(title=fr"Adaboost train set, noise={noise}"))

    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # fit_and_evaluate_adaboost(NO_NOISE)
    fit_and_evaluate_adaboost(NOISE)
