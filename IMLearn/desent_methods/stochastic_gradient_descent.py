from __future__ import annotations
from typing import Callable, Tuple
import numpy as np

from IMLearn.base import BaseModule, BaseLR
from .gradient_descent import default_callback
from .learning_rate import FixedLR


class StochasticGradientDescent:
    """
    Gradient Descent algorithm

    Attributes:
    -----------
    learning_rate_: BaseLR
        Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

    tol_: float
        The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
        specified tolerance

    max_iter_: int
        The maximum number of GD iterations to be performed before stopping training

    callback_: Callable[[...], None], default=default_callback
        A callable function to be called after each update of the model while fitting to given data.
        Callable function receives as input any argument relevant for the current GD iteration. Arguments
        are specified in the `GradientDescent.fit` function
    """

    def __init__(self,
                 learning_rate: BaseLR = FixedLR(1e-3),
                 tol: float = 1e-5,
                 max_iter: int = 1000,
                 batch_size: int = 1,
                 callback: Callable[[...], None] = default_callback):
        """
        Instantiate a new instance of the GradientDescent class

        Parameters
        ----------
        learning_rate: BaseLR, default=FixedLR(1e-3)
            Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

        tol: float, default=1e-5
            The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
            specified tolerance

        max_iter: int, default=1000
            The maximum number of GD iterations to be performed before stopping training

        batch_size: int, default=1
            Number of samples to randomly select at each iteration of the SGD algorithm

        callback: Callable[[...], None], default=default_callback
            A callable function to be called after each update of the model while fitting to given data.
            Callable function receives as input any argument relevant for the current GD iteration. Arguments
            are specified in the `GradientDescent.fit` function
        """
        self.learning_rate_ = learning_rate
        self.tol_ = tol
        self.max_iter_ = max_iter
        self.batch_size_ = batch_size
        self.callback_ = callback

    def fit(self, f: BaseModule, X: np.ndarray, y: np.ndarray):
        """
        Optimize module using SGD iterations over given input samples and responses

        Parameters
        ----------
        f : BaseModule
        Objective function (module) to be minimized by SGD


        X : ndarray of shape (n_samples, n_features)
            Input data to optimize module over

        y : ndarray of shape (n_samples, )
            Responses of input data to optimize module over

        Returns
        -------
        solution: ndarray of shape (n_features)
            Obtained solution for module optimization

        Notes
        -----
        - Optimization is performed as long as self.max_iter_ has not been reached and that
        Euclidean norm of w^(t)-w^(t-1) is more than the specified self.tol_

        - At each iteration the learning rate is specified according to self.learning_rate_.lr_step

        - At the end of each iteration the self.callback_ function is called passing self and the
        following named arguments:
            - solver: GradientDescent
                self, the current instance of GradientDescent
            - weights: ndarray of shape specified by module's weights
                Current weights of objective
            - val: ndarray of shape specified by module's compute_output function
                Value of objective function at current point, over given data X, y
            - grad:  ndarray of shape specified by module's compute_jacobian function
                Module's jacobian with respect to the weights and at current point, over given data X, y
            - t: int
                Current GD iteration
            - eta: float
                Learning rate used at current iteration
            - delta: float
                Euclidean norm of w^(t)-w^(t-1)
            - batch_indices: np.ndarray of shape (n_batch,)
                Sample indices used in current SGD iteration
        """
        raise NotImplementedError()

    def _partial_fit(self, f: BaseModule, X: np.ndarray, y: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform a SGD iteration over given samples

        Parameters
        ----------
        f : BaseModule
        Objective function (module) to be minimized by SGD

        X : ndarray of shape (n_batch, n_features)
            Input data to optimize module over

        y : ndarray of shape (n_batch, )
            Responses of input data to optimize module over

        t: int
            Current SGD iteration

        Returns
        -------
        val: ndarray of shape (n_features,)
            Value of objective optimized, at current position, based on given samples

        jac: ndarray of shape (n_features, )
            Jacobian on objective optimized, at current position, based on given samples

        eta: float
            learning rate used at current iteration
        """
        iteration = 1
        x_t = f.weights
        x_t_1 = x_t
        average = x_t
        best_y = f.compute_output(X=X, y=y)
        best_x = x_t
        # todo: maybe < and not <=
        # todo: x_t is unnecessary- f.weights can be used instead.
        while iteration <= self.max_iter_:
            x_t_1 = x_t - self.learning_rate_.lr_step(t=iteration) * f.compute_jacobian(X=X, y=y)
            f.weights = x_t_1

            # average:
            average += x_t_1.copy()
            # best:
            if f.compute_output(X=X, y=y) < best_y:
                best_y = f.compute_output(X=X, y=y)
                best_x = x_t_1

            # callback
            self.callback_(weights=x_t_1, values=f.compute_output(X=X, y=y))

            if np.linalg.norm(x_t - x_t_1) < self.tol_:
                break
            x_t = x_t_1
            iteration += 1

        return x_t_1