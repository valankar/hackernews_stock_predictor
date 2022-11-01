#!/usr/bin/env python
"""Train and predict."""

from os.path import exists
from pathlib import Path

import pandas as pd
from joblib import dump, load
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STORAGE_DIR = f'{Path.home()}/code/predictor/storage'
PREDICTIONS = f'{STORAGE_DIR}/predictions.pkl.gz'
LOAD_MODEL = False
PRINT_ONLY = False


def load_data():
    """Load hackernews and stocks data."""
    hackernews_df = pd.read_pickle(f'{STORAGE_DIR}/hackernews.pkl.gz')
    stocks_df = pd.read_pickle(f'{STORAGE_DIR}/stocks.pkl.gz')
    return (hackernews_df, stocks_df)


def vectorize(dataframe):
    """Hashing vectorize dataframe by day."""
    vectorizer = HashingVectorizer(
        n_features=100000, lowercase=False, tokenizer=lambda x: [x])
    day_dfs = []
    for day in sorted(dataframe.index.unique()):
        # Duplicate all rows based on their count.
        new_df = dataframe.loc[day:day].reset_index()
        new_df = new_df.loc[new_df.index.repeat(new_df['count'].astype(int))].drop(
            columns='count').set_index('date')
        transformed = vectorizer.transform(new_df['gram'])
        day_df = pd.DataFrame(transformed.sum(axis=0), index=[day])
        day_dfs.append(day_df)
    final_df = pd.concat(day_dfs)
    return final_df


def prepare_dfs(up_to=None):
    """Sanitize dfs for training."""
    hackernews_df, stocks_df = load_data()
    # Increment days in hackernews, which will be matched with stocks for that day.
    # i.e. news from day before predicts stocks of current day
    hackernews_df.index = hackernews_df.index + pd.Timedelta(1, unit='D')
    # Restrict training data.
    if up_to:
        print(f'Making model up to {up_to}')
        hackernews_df = hackernews_df.loc[:up_to]
    # Restrict training data to dates with known data in stocks.
    hackernews_df = hackernews_df.loc[hackernews_df.index.isin(
        stocks_df.index.unique())]
    stocks_df = stocks_df.loc[stocks_df.index.isin(
        hackernews_df.index.unique())]
    stocks_df = stocks_df['regular_market_change_percent']
    return vectorize(hackernews_df), stocks_df


def make_model(up_to=None, save_model=False, hackernews_df=None, stocks_df=None):
    """Make prediction model."""
    if hackernews_df is None or stocks_df is None:
        hackernews_df, stocks_df = prepare_dfs(up_to)
    clf = make_pipeline(StandardScaler(), LinearRegression(n_jobs=-1))
    clf.fit(hackernews_df, stocks_df)
    if save_model and not PRINT_ONLY:
        dump(clf, f'{STORAGE_DIR}/model.joblib.gz')
    return clf


def main():
    """Main."""
    if LOAD_MODEL:
        clf = load(f'{STORAGE_DIR}/model.joblib.gz')
    else:
        clf = make_model(save_model=True)
    hackernews_df, stocks_df = load_data()
    stocks_df = stocks_df['regular_market_change_percent']

    yesterday = pd.Timestamp.now() - pd.Timedelta(1, unit='D')
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    predictions = clf.predict(
        vectorize(hackernews_df.loc[yesterday_str:yesterday_str]))
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
