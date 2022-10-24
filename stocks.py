#!/usr/bin/env python
"""Download stock data into dataframe."""

import subprocess
from datetime import datetime
from os.path import exists
from pathlib import Path

import pandas as pd


DOWNLOAD_DIR = f'{Path.home()}/code/predictor/downloads'
STORAGE_DIR = f'{Path.home()}/code/predictor/storage'
STOCKS_TIMESTAMP = f'{STORAGE_DIR}/stocks-timestamp.txt'
STEAMPIPE = f'{Path.home()}/bin/steampipe'

TICKERS = (
    'SCHK',
    'SCHB',
    'SCHX',
    'SCHG',
    'SCHV',
    'SCHD',
    'SCHM',
    'SCHA',
    'SCHH',
    'SCHY',
    'SCHC',
    'SCHF',
    'SCHE',
    'SCHJ',
    'SCHI',
    'SCHZ',
    'SCHP',
    'SCHO',
    'SCHR',
    'SCHQ',
    'FNDB',
    'FNDX',
    'FNDA',
    'FNDF',
    'FNDC',
    'FNDE',
    'GC=F',
    'SI=F',
)


def get_stock_data():
    """Get latest stock prices."""
    today_str = datetime.now().strftime('%Y-%m-%d')
    output_file = f'{DOWNLOAD_DIR}/stocks-{today_str}.csv'
    if exists(output_file):
        return output_file
    subprocess.run(
        # pylint: disable=line-too-long
        f'''{STEAMPIPE} query "select regular_market_time, symbol, regular_market_price, regular_market_change_percent \
            from finance_quote where symbol in ({','.join([f"'{x}'" for x in TICKERS])}) order by symbol asc" --output csv > {output_file}''',
        shell=True, check=True)
    return output_file


def main():
    """Main."""
    stocks_df = pd.read_csv(
        get_stock_data(), index_col=0, parse_dates=True, infer_datetime_format=True)
    # Remove time component
    stocks_df.index = pd.DatetimeIndex(stocks_df.index.strftime('%Y-%m-%d'))
    stocks_df = stocks_df.pivot(columns='symbol')
    stocks_df = stocks_df.dropna()
    storage_file = f'{STORAGE_DIR}/stocks.pkl.gz'
    if exists(storage_file):
        # Merge old data
        stocks_df = pd.concat(
            [pd.read_pickle(storage_file), stocks_df]
        )
        stocks_df = stocks_df[~stocks_df.index.duplicated(keep='last')]
    stocks_df.to_pickle(storage_file, compression='gzip')


if __name__ == '__main__':
    main()
