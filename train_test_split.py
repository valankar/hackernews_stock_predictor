#!/usr/bin/env python
"""Training test."""

import math

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import predict


def calculate():
    """Calculate MSE and RMSE."""
    hackernews_df, stocks_df = predict.prepare_dfs()
    x_train, x_test, y_train, y_test = train_test_split(
        hackernews_df, stocks_df,
    )
    clf = predict.make_model(hackernews_df=x_train, stocks_df=y_train)
    prediction = clf.predict(x_test)
    mean_squared_score = mean_squared_error(y_test, prediction)
    return (mean_squared_score, math.sqrt(mean_squared_score))


def main():
    """Main."""
    mse, rmse = calculate()
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Square Error: {rmse}')


if __name__ == '__main__':
    main()
