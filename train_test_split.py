#!/usr/bin/env python
"""Training test."""

import math

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import predict


def main():
    """Main."""
    hackernews_df, stocks_df = predict.prepare_dfs()
    x_train, x_test, y_train, y_test = train_test_split(
        hackernews_df, stocks_df, test_size=0.2,
    )
    clf = LinearRegression()
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    mean_squared_score = mean_squared_error(y_test, prediction)
    print(f'Mean Squared Error: {mean_squared_score}')
    print(f'Root Mean Square Error: {math.sqrt(mean_squared_score)}')


if __name__ == '__main__':
    main()
