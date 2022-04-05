import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

# IMAGE_PATH = r"C:\Users\eviatar\Desktop\eviatar\Study\YearD\semester b\I.M.L\repo\IML.HUJI\plots\ex2\CityTemperature\\"
DATA_PATH = r"../datasets/City_Temperature.csv"

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # -creating data frame:
    temp_data = pd.read_csv(filename, parse_dates=['Date'])
    temp_data["DayOfYear"] = temp_data["Date"].dt.dayofyear
    # days_filter = temp_data['DayOfYear'] <= 365

    # filtering the dominance noise (such degrees are possible but not in the 4 mentioned countries:
    temperature_filter = temp_data['Temp'] > -50
    temp_data = temp_data.loc[temperature_filter]
    # print(temp_data.min(axis=0))
    # print("------------------------")
    # print(temp_data.max(axis=0))
    # print(temp_data)
    return temp_data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and pre-processing of city temperature dataset
    data = load_data(DATA_PATH)

    # Question 2 - Exploring data for specific country
    # raise NotImplementedError()
    israel_data = data.drop(data[data["Country"] != "Israel"].index)
    israel_data["Year"] = israel_data["Year"].astype(str)
    figure1 = px.scatter(israel_data, x="DayOfYear", y="Temp", color="Year",
                         title="Average daily temperature as a function of the DayOfYear in Israel.")
    # figure1.write_image(IMAGE_PATH + "Israel_temperature.png")
    figure1.show()
    monthly_data_il = israel_data.groupby(['Month'], as_index=False).agg({'Temp': ['std']})
    figure2 = px.bar(x=monthly_data_il['Month'], y=monthly_data_il['Temp']['std'],
                     title="Standard deviation of the daily temperatures in Israel:",
                     labels={'x': 'Month', 'y': 'Standard deviation'})
    # figure2.write_image(IMAGE_PATH + "Monthly_temperature_Il.png")
    figure2.show()

    # Based on this graph, we expect that the model will perform  better in different times in the year.
    # I.e. in month lower the std is the better the model will preform.
    # (Since the Std is the square root from the mean and the data is sampled uniformly)

    # Question 3 - Exploring differences between countries
    data_by_month = data.groupby(['Month', 'Country'])["Temp"].agg(['mean', 'std']).reset_index()
    figure3 = px.line(data_by_month, x='Month', y='mean', error_y='std', color='Country',
                      title='Average and standard deviation of the temperature by country and by month:',
                      labels={'x': 'Month', 'y': 'Mean std as error'})
    # figure3.write_image(IMAGE_PATH + "Monthly_temperature.png")
    figure3.show()
    # Based on this graph, not all the countries share a similar pattern we can see that clearly from the
    # maximum and minimum pikes, as well as the slopes in the different areas.
    # e.g. the South Africa graph is almost inverse in those aspects to Israel's and Jordan's graphs.
    # Jordan is the country that the Israel's model will work well.
    # (Based on the error and the similarity between the graphs).
    # About the Netherlands we can expect for some similarity and coherency
    # but since the translation difference and the fact that error are quite different
    # we can't expect too much from that.
    # About South Africa the Model will not work well, based on the error and on the graphs.

    # Question 4 - Fitting model for different values of `k`
    train_x, train_y, test_x, test_y = split_train_test(israel_data['DayOfYear'], israel_data["Temp"])
    loss = []
    for k in range(1, 11):
        poly_model = PolynomialFitting(k)
        poly_model.fit(train_x.to_numpy(), train_y.to_numpy())
        k_loss = round(poly_model.loss(test_x.to_numpy(), test_y.to_numpy()), 2)
        print(f"Fitting polynomial with degree- {k} yields loss of- {k_loss}")
        loss.append(k_loss)
    figure4 = px.bar(x=range(1, 11), y=loss, title='Test error for each value of degree.',
                     labels={'x': "Polynomial degree", 'y': 'Loss'})
    # figure4.write_image(IMAGE_PATH + "Loss_degree.png")
    figure4.show()
    # Based on this graph and Based of the Loss estimation polynomial with degree 5 best fits the data.
    # Yes, this estimator is one way to estimate the fitting.

    # Question 5 - Evaluating fitted model on different countries
    best_model = PolynomialFitting(5)
    best_model.fit(israel_data["DayOfYear"], israel_data["Temp"])

    errors = []
    countries = data["Country"].unique()
    countries = np.delete(countries, np.where(countries == "Israel"))
    for country in countries:
        country_data = data[data["Country"] == country]
        errors.append(best_model.loss(country_data["DayOfYear"], country_data['Temp']))
    errors = np.round(errors, 2)

    figure5 = px.bar(x=countries, y=errors, labels={"x": "country", "y": "loss"},
                     title="Loss after fitting of each country.")
    # figure5.write_image(IMAGE_PATH + "Loss.png")
    figure5.show()


