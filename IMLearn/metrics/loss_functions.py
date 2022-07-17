import numpy as np


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    # raise NotImplementedError()
    return np.sum((y_true - y_pred) ** 2) / y_true.shape[0]


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    # raise NotImplementedError()
    factor = y_true.shape[0]
    if not normalize:
        factor = 1
    return np.count_nonzero(y_pred != y_true) / factor


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    # The Accuracy is the number of correct classification out of all predictions: (T P+T N)/(P+N)
    return np.sum(y_pred == y_true) / len(y_true)


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    # Naively computes log(s_i) even when y_i = 0
    # return -y.dot(np.log(s))

    # samples = len(y_true)
    #
    # correct_log_probs = -np.log(y_pred[range(samples), y_true])
    # data_loss = np.sum(correct_log_probs) / samples
    #
    # return data_loss

    loss = -np.sum(y_true * np.log(y_pred))
    return loss / float(y_pred.shape[0])


def softmax(X: np.ndarray) -> np.ndarray:
    """
    Compute the Softmax function for each sample in given data

    Parameters:
    -----------
    X: ndarray of shape (n_samples, n_features)

    Returns:
    --------
    output: ndarray of shape (n_samples, n_features)
        Softmax(x) for every sample x in given data X
    """
    # todo: check if Softmax(x) for every sample x in given data X
    exp_scores = np.exp(X)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs
