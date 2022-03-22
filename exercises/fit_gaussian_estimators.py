import plotly
import plotly.express as px
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # ------------------------------------------------
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, size=1000)
    uni_gau = UnivariateGaussian().fit(samples)
    # todo check if this print is ok
    print((uni_gau.mu_, uni_gau.var_))

    # Question 2 - Empirically showing sample mean is consistent

    # todo maybe range (10, 1010, 10)
    ms = np.linspace(10, 1000, num=100).astype(int)
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
                               height=600)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    px.scatter(x=samples, y=uni_gau.pdf(samples),
               title="Empirical PDF",
               labels=dict(x="sample values", y="pdf")).show()
    # As expected we've got a gaussian distribution centered in the mean = 10 and with relatively low variance = 1

    # Ex1 Quiz:
    # a = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
    #               -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    #
    # b = UnivariateGaussian.log_likelihood(1, 1, a)
    # c = UnivariateGaussian.log_likelihood(10, 1, a)


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = [0, 0, 4, 0]
    covariance = [[1, 0.2, 0, 0.5],
                  [0.2, 2, 0, 0],
                  [0, 0, 1, 0],
                  [0.5, 0, 0, 1]]

    X = np.random.multivariate_normal(mu, covariance, 1000)

    mul_gau = MultivariateGaussian()
    mul_gau.fit(X)
    print(np.round(mul_gau.mu_, 3))
    print(np.round(mul_gau.cov_, 3))

    # Question 5 - Likelihood evaluation
    # raise NotImplementedError()

    # Question 6 - Maximum likelihood
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
