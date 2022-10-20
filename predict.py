#!/usr/bin/env python
"""Train and predict."""

from os.path import exists
from pathlib import Path

import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing


STORAGE_DIR = f'{Path.home()}/code/predictor/storage'
PLOT_DIR = f'{Path.home()}/code/predictor/web'
PREDICTIONS = f'{STORAGE_DIR}/predictions.pkl.gz'
LOAD_MODEL = False
PRINT_ONLY = False


def load_data():
    """Load hackernews and stocks data."""
    hackernews_df = pd.read_pickle(f'{STORAGE_DIR}/hackernews.pkl.gz')
    stocks_df = pd.read_pickle(f'{STORAGE_DIR}/stocks.pkl.gz')
    return (hackernews_df, stocks_df)


def make_model():
    """Make prediction model."""
    hackernews_df, stocks_df = load_data()
    # Need to create columns for all data, including predicted time.
    hackernews_df = hackernews_df.pivot(columns='gram').fillna(0)
    # Increment days in hackernews, which will be matched with stocks for that day.
    # i.e. news from day before predicts stocks of current day
    hackernews_df.index = hackernews_df.index + pd.Timedelta(1, unit='D')
    # Earliest known good data.
    hackernews_df = hackernews_df.loc['2022-10-13':]
    # Restrict training data to dates with known data in stocks.
    hackernews_df = hackernews_df.loc[hackernews_df.index.isin(
        stocks_df.index)]
    stocks_df = stocks_df.loc[stocks_df.index.isin(
        hackernews_df.index.unique())]
    stocks_df = stocks_df['regular_market_change_percent']

    multi_output_clf = LinearRegression()
    multi_output_clf.fit(preprocessing.scale(hackernews_df, axis=1), stocks_df)
    if not PRINT_ONLY:
        dump(multi_output_clf, f'{STORAGE_DIR}/model.joblib.gz')
    return multi_output_clf


def main():
    """Main."""
    if LOAD_MODEL:
        multi_output_clf = load(f'{STORAGE_DIR}/model.joblib.gz')
    else:
        multi_output_clf = make_model()
    hackernews_df, stocks_df = load_data()
    stocks_df = stocks_df['regular_market_change_percent']
    hackernews_df = hackernews_df.pivot(columns='gram').fillna(0)

    yesterday = pd.Timestamp.now() - pd.Timedelta(1, unit='D')
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    predictions = multi_output_clf.predict(
        preprocessing.scale(hackernews_df.loc[yesterday_str:yesterday_str], axis=1))
    new_df = pd.DataFrame(
        columns=stocks_df.columns, data=predictions, index=[pd.Timestamp(today_str)])
    if exists(PREDICTIONS):
        # Remove any old prediction dates which we don't have prices for.
        # These might be weekends, etc.
        old_df = pd.read_pickle(PREDICTIONS)
        old_df = old_df.loc[old_df.index.isin(stocks_df.index)]
        predictions_df = pd.concat([old_df, new_df])
    else:
        predictions_df = new_df
    predictions_df.index.name = 'date'
    predictions_df.columns.name = 'symbol'
    # Remove any duplicate dates, keeping last.
    predictions_df = predictions_df[~predictions_df.index.duplicated(
        keep='last')]
    if PRINT_ONLY:
        print(predictions_df[::-1])
    else:
        predictions_df.to_pickle(PREDICTIONS)


if __name__ == '__main__':
    main()
