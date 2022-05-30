from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    x = np.linspace(-1.2, 2, n_samples)
    f = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    eps = np.random.normal(0, noise, n_samples)
    y = f + eps

    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(x), pd.Series(y), 2 / 3)

    X_train = X_train.to_numpy()[:, 0]
    X_test = X_test.to_numpy()[:, 0]
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    fig = go.Figure(
        [go.Scatter(x=x, y=f, mode='lines+markers', marker=dict(color="black"), name=r'Noiseless Model'),
         go.Scatter(x=X_train, y=y_train, mode='markers', name=r'Train samples'),
         go.Scatter(x=X_test, y=y_test, mode='markers', name=r'Test samples')])
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="y")
    fig.update_layout(title_text=rf"Generated a dataset of {n_samples} samples and noise={noise}")
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    average_training_and_validation_errors = [cross_validate(PolynomialFitting(k), X_train, y_train, mean_square_error)
                                              for k in range(11)]
    average_training_errors = [tup[0] for tup in average_training_and_validation_errors]
    average_validation_errors = [tup[1] for tup in average_training_and_validation_errors]
    ks = [k for k in range(11)]

    fig = go.Figure(
        [go.Scatter(x=ks, y=average_training_errors, mode='markers', marker=dict(color="black"),
                    name=fr'Average training errors'),
         go.Scatter(x=ks, y=average_validation_errors, mode='markers', name=r'Average validation errors samples')])

    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="y")
    fig.update_layout(title_text=f"5-fold cross-validation for each of the polynomial degrees k = 0,1,...,10\n"
                                 f" As a function of training- and validation errors\n"
                                 f"({n_samples} samples and noise={noise})")
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k = np.argmin(average_validation_errors)
    est = PolynomialFitting(k)
    est.fit(X_train, y_train)
    test_error = mean_square_error(y_test, est.predict(X_test))
    print(f"\nFor a Model with {n_samples} samples and noise={noise}:")
    print(f"{k} is the polynomial degree for which the lowest validation error was achieved")
    print(f"polynomial degree with {k} got MSE= {np.round(test_error, 2)}.")
    print(f"polynomial degree with {k}  got Validation MSE= {np.round(average_validation_errors[k], 2)}.")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    X_train, y_train, X_test, y_test = split_train_test(X, y, n_samples / X.shape[0])
    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions

    # Ridge:

    lam = np.linspace(0.02, 2.02, num=n_evaluations)

    lasso_average_training_and_validation_errors = [
        cross_validate(RidgeRegression(k), X_train.to_numpy(), y_train.to_numpy(), mean_square_error)
        for k in lam]
    ridge_training_errors = [tup[0] for tup in lasso_average_training_and_validation_errors]
    ridge_validation_errors = [tup[1] for tup in lasso_average_training_and_validation_errors]

    fig = go.Figure([go.Scatter(x=lam, y=ridge_training_errors, mode='lines+markers', name=r'Average Train Error'),
                     go.Scatter(x=lam, y=ridge_validation_errors, mode='lines+markers',
                                name=r'Average Validation Error')])
    fig.update_xaxes(title_text="regularization parameter")
    fig.update_yaxes(title_text="error values")
    fig.update_layout(title_text=rf"Ridge regularization:")
    fig.show()

    # Lasso:
    lasso_average_training_and_validation_errors = [
        cross_validate(Lasso(k), X_train.to_numpy(), y_train.to_numpy(), mean_square_error)
        for k in lam]
    lasso_training_errors = [tup[0] for tup in lasso_average_training_and_validation_errors]
    lasso_validation_errors = [tup[1] for tup in lasso_average_training_and_validation_errors]

    fig = go.Figure([go.Scatter(x=lam, y=lasso_training_errors, mode='lines+markers', name=r'Average Train Error'),
                     go.Scatter(x=lam, y=lasso_validation_errors, mode='lines+markers',
                                name=r'Average Validation Error')])
    fig.update_xaxes(title_text="regularization parameter")
    fig.update_yaxes(title_text="error values")
    fig.update_layout(title_text=rf"Lasso regularization:")
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    lasso_best_lam = lam[np.argmin(lasso_validation_errors)]
    ridge_best_lam = lam[np.argmin(ridge_validation_errors)]

    linear_regression = LinearRegression()
    ridge = RidgeRegression(ridge_best_lam)
    lasso = Lasso(alpha=lasso_best_lam)
    linear_regression.fit(X_train.to_numpy(), y_train.to_numpy())
    ridge.fit(X_train.to_numpy(), y_train.to_numpy())
    lasso.fit(X_train.to_numpy(), y_train.to_numpy())

    print(f"{ridge_best_lam}"
          f" as regularization parameter value achieved the best validation errors for the Ridge regularizations ")
    print(f"{lasso_best_lam}"
          f" as regularization parameter value achieved the best validation errors for the Lasso regularizations ")

    print(f"Ridge fitted with best Lambda= {ridge_best_lam} emits error: {ridge.loss(X_test.to_numpy(), y_test.to_numpy())}")
    lasso_error = mean_square_error(y_test.to_numpy(), lasso.predict(X_test.to_numpy()))
    print(f"Lasso fitted with best Lambda= {ridge_best_lam} emits error: {lasso_error}")
    print(f"Least Squares emits error: {linear_regression.loss(X_test.to_numpy(), y_test.to_numpy())}")


if __name__ == '__main__':
    np.random.seed(0)
    # Q1-Q3:
    select_polynomial_degree()
    # Q4:
    select_polynomial_degree(noise=0)
    # Q5:
    select_polynomial_degree(n_samples=1500, noise=10)
    # Q6 - Q8:
    select_regularization_parameter()
