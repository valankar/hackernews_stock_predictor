#!/usr/bin/env python
"""Plot predictions."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

import predict

STORAGE_DIR = f'{Path.home()}/code/predictor/storage'
PLOT_DIR = f'{Path.home()}/code/predictor/web'
COLOR_GREEN = '#3d9970'
COLOR_RED = '#ff4136'


def plot_predictions():
    """Plot predictions."""
    today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    predictions_df = pd.read_pickle(predict.PREDICTIONS)
    predictions_df.index.name = 'regular_market_time'
    graph_df = predictions_df.loc[today_str:today_str]
    hackernews_df, stocks_df = predict.load_data()
    yesterday_str = (pd.Timestamp.now() - pd.Timedelta(1,
                     unit='D')).strftime('%Y-%m-%d')

    values = graph_df.values[0]
    fig = go.Figure(go.Bar(
        x=graph_df.columns,
        y=values,
        marker_color=[COLOR_GREEN if x > 0 else COLOR_RED for x in values],
    ))
    fig.update_layout(title=f'Predictions for {today_str}')
    fig.update_yaxes(title_text='Percent')
    fig.update_xaxes(title_text='ETF')

    with open(f'{PLOT_DIR}/index.html', 'w', encoding='utf-8') as output_file:
        output_file.write(fig.to_html(full_html=True, include_plotlyjs='cdn'))
        output_file.write('<h2>Historical Percent Change</h2>')
        stocks_df['regular_market_change_percent'].tail(30).iloc[::-1].to_html(
            output_file, float_format='%.2f')
        output_file.write(
            '<h2>Predictions Based On Linear Regression Model</h2>')
        predictions_df.tail(
            30).iloc[::-1].to_html(output_file, float_format='%.2f')
        output_file.write(
            f'<h2>Hackernews top phrases in comments for {yesterday_str}</h2>')
        hackernews_df.loc[yesterday_str].groupby(['gram']).sum().sort_values(
            by='count', ascending=False).head(30).to_html(output_file, float_format='%.0f')


def main():
    """Main."""
    plot_predictions()


if __name__ == '__main__':
    main()
