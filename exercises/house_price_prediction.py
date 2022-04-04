from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

HOUSE_DATA = r"../datasets/house_prices.csv"

# todo: to comment
IMAGE_PATH = r"C:\Users\eviatar\Desktop\eviatar\Study\YearD\semester b\I.M.L\repo\IML.HUJI\plots\ex2\house\\"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # -creating data frame:
    data = pd.read_csv(filename)
    # -omits id column as its a clear redundant noise:
    data = data.drop(['id'], axis=1)
    # -dealing with nulls (since  data.isnull().sum() is very low we will drop them):
    data = data.dropna()

    # dealing with samples that has negative prices or houses that are too small
    data = data[(data["sqft_living"] > 15)]
    data = data[(data["price"] > 0)]

    # replace the date with One Hot representation of month and year:
    data['date'] = pd.to_datetime(data['date'])
    # todo: switch day and month position
    data['date'] = data['date'].dt.year.astype(str) + data['date'].dt.month.astype(str)
    data = pd.get_dummies(data=data, columns=['date'])
    # dealing Zip code by replacing it with One Hot representation:
    data = pd.get_dummies(data=data, columns=['zipcode'])

    # dealing with feature that has a significant low correlation after plotting the heatmap.
    data = data.drop(["yr_built"], axis=1)
    # features deduction

    # treating invalid/ missing values

    y = data['price']
    data.drop(['price'], axis=1, inplace=True)

    return data, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # corr = X.corr()
    # corr.style.background_gradient(cmap='coolwarm').set_precision(2)
    for i, column in enumerate(X.columns):
        cov = pd.Series.cov(X.iloc[:, i], y)
        std = pd.Series.std(X.iloc[:, i]) * pd.Series.std(y)
        correlation = cov / std

        plot = px.scatter(x=X.iloc[:, i], y=y, trendline="ols", trendline_color_override="black",
                          title=f"Pearson Correlation between {column}"
                                f" and response:\n [P.C = {np.round(correlation, 2)}]",
                          labels=dict(x=column + " values", y="price"))
        feature_name = str(column)
        output_path = IMAGE_PATH
        plot.write_image(output_path + feature_name + "correlation.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocess of housing prices dataset
    x, y = load_data(HOUSE_DATA)

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(x, y)
    # raise NotImplementedError()

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(x, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    means = []
    stds = []
    linear_reg = LinearRegression()
    for p in range(10, 101, 1):
        p_losses = []
        for _ in range(10):
            train_proportion = p / 100
            px_train, py_train, _, _ = split_train_test(train_x, train_y, train_proportion)
            linear_reg.fit(px_train.to_numpy(), py_train.to_numpy())
            # todo why not with the matching test proportion ?
            loss = linear_reg.loss(test_x.to_numpy(), test_y.to_numpy())
            p_losses.append(loss)
        means.append(np.mean(p_losses))
        stds.append(np.std(p_losses))
    means = np.asarray(means)
    stds = np.asarray(stds)

    figure = go.Figure([go.Scatter(x=np.arange(10, 101), y=means, mode='lines'),
                        go.Scatter(x=np.arange(10, 101), y=means + 2 * stds, showlegend=False,
                                   marker=dict(color='lightgrey'), name="Confidence"),
                        go.Scatter(x=np.arange(10, 101), y=means - 2 * stds, fill='tonexty', showlegend=False,
                                   mode="lines", marker=dict(color='lightgrey'), name="Confidence")])
    figure.update_xaxes(title_text="training-set's percents")
    figure.update_yaxes(title_text="test-set's loss")
    figure.update_layout(title_text="Loss as function of training size")
    figure.write_image(IMAGE_PATH + "mean.png")
    figure.show()
