from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"

# todo check that:
import sys

sys.path.append(r'\C:\Users\eviatar\Desktop\eviatar\Study\YearD\semester b\I.M.L\repo\IML.HUJI')
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    # raise NotImplementedError()
    samples = np.random.normal(10, 1, size=1000)
    uni_gau = UnivariateGaussian().fit(samples)
    # todo check if this print is ok
    print((uni_gau.mu_, uni_gau.var_))

    # Question 2 - Empirically showing sample mean is consistent
    # raise NotImplementedError()

    # todo maybe range (10, 1010, 10)
    ms = np.linspace(10, 1000, num=100).astype(np.int)
    estimated_diff = []
    for m in ms:
        m_samples = np.random.normal(10, 1, size=m)
        m_uni_gau = UnivariateGaussian().fit(m_samples)
        estimated_diff.append(np.abs(10 - m_uni_gau.mu_))

    go.Figure([go.Scatter(x=ms, y=estimated_diff, mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{The absolute distance between the estimated-"
                                     r" and true value of the expectation As Function Of Number Of Samples}$",
                               xaxis_title=r"$m\text{ - number of samples}$",
                               yaxis_title=r"$|\hat\mu - \mu|$",
                               # yaxis_title=r"r$\hat\mu$ ",
                               height=600)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    # raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
