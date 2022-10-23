#!/usr/bin/env python
"""Re-run all predictions."""

import pandas as pd

import predict


def main():
    """Main."""
    hackernews_df, stocks_df = predict.load_data()
    stocks_df = stocks_df['regular_market_change_percent']
    hackernews_df = hackernews_df.loc['2022-10-13':]
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    prediction_dfs = []
    for day in sorted(hackernews_df.index.unique()):
        day_str = day.strftime('%Y-%m-%d')
        next_day = day + pd.Timedelta(1, unit='D')
        if day_str == today_str:
            break
        multi_output_clf = predict.make_model(up_to=day_str)
        prediction = multi_output_clf.predict(
            predict.vectorize(hackernews_df.loc[day_str:day_str]))
        prediction_dfs.append(pd.DataFrame(
            columns=stocks_df.columns, data=prediction, index=[next_day]))

    prediction_df = pd.concat(prediction_dfs)
    prediction_df = pd.concat([
        prediction_df.loc[prediction_df.index.isin(stocks_df.index)],
        prediction_df.iloc[-1:]]).sort_index()

    # Remove any duplicate dates, keeping last.
    prediction_df = prediction_df[~prediction_df.index.duplicated(
        keep='last')]
    print(prediction_df[::-1])
    prediction_df.to_pickle(predict.PREDICTIONS)


if __name__ == '__main__':
    main()
