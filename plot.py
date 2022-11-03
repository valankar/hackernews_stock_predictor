#!/usr/bin/env python
"""Plot predictions."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

import predict
import train_test_split

STORAGE_DIR = f'{Path.home()}/code/predictor/storage'
PLOT_DIR = f'{Path.home()}/code/predictor/web'
COLOR_GREEN = '#3d9970'
COLOR_RED = '#ff4136'

HTML_PRE = '''
<html>
  <link rel="stylesheet" type="text/css" href="df_style.css"/>
  <body>
'''

HTML_POST = '''
  </body>
</html>.
'''


def write_table(output_file, dataframe, float_format):
    """Output table with styling."""
    dataframe.to_html(output_file, classes='mystyle',
                      float_format=float_format)


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

    mse, rmse = train_test_split.calculate()

    with open(f'{PLOT_DIR}/index.html', 'w', encoding='utf-8') as output_file:
        output_file.write(HTML_PRE)
        output_file.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        output_file.write('<h2>Historical Percent Change</h2>')
        write_table(
            output_file,
            stocks_df['regular_market_change_percent'].tail(30).iloc[::-1],
            float_format='%.2f')
        output_file.write(
            '<h2>Predictions Based On Linear Regression Model</h2>')
        write_table(
            output_file,
            predictions_df.tail(30).iloc[::-1],
            float_format='%.2f')
        output_file.write(
            f'<h2>Hackernews top phrases in comments for {yesterday_str}</h2>')
        write_table(
            output_file,
            hackernews_df.loc[yesterday_str].groupby(['gram']).sum().sort_values(
                by='count', ascending=False).head(30),
            float_format='%.0f')
        output_file.write('<h2>Error</h2>')
        output_file.write('<pre>')
        output_file.write(f'Mean Squared Error: {mse:.3f}\n')
        output_file.write(f'Root Mean Square Error: {rmse:.3f}\n')
        output_file.write('</pre>')
        output_file.write(HTML_POST)


def main():
    """Main."""
    plot_predictions()


if __name__ == '__main__':
    main()
