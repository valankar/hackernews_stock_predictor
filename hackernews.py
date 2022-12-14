#!/usr/bin/env python
"""Download hackernews data into dataframe."""

import glob
import re
import subprocess
import warnings
from datetime import datetime
from os.path import exists
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from nltk import ngrams
from nltk.corpus import stopwords

DOWNLOAD_DIR = f'{Path.home()}/code/predictor/downloads'
STORAGE_DIR = f'{Path.home()}/code/predictor/storage'
STORAGE_FILE = f'{STORAGE_DIR}/hackernews.pkl.gz'
HACKERNEWS_TIMESTAMP = f'{STORAGE_DIR}/hackernews-timestamp.txt'
STEAMPIPE = f'{Path.home()}/bin/steampipe'
NGRAMS = (2, 3, 4)
IMPORT_ALL = False

STOPWORDS = [word.lower() for word in stopwords.words('english')]


def download_hackernews():
    """Download latest hackernews data."""
    today_str = datetime.now().strftime('%Y-%m-%d')
    output_file = f'{DOWNLOAD_DIR}/hackernews-{today_str}.csv'
    if exists(output_file) and Path(output_file).stat().st_size > 0:
        return output_file
    subprocess.run(
        # pylint: disable-next=line-too-long
        f'''{STEAMPIPE} query "select time, text  from hackernews_item where type='comment' order by time asc" --output csv > {output_file}''',
        shell=True, check=True)
    return output_file


def sum_duplicates(dataframe):
    """Sum duplicate entries by date and gram."""
    new_df = dataframe.groupby(['gram', dataframe.index]).sum()
    unstack_df = new_df.unstack(level=1)
    day_dfs = []
    for column in unstack_df.columns:
        # column looks like ('count', '2022-10-11')
        reset = unstack_df[column].reset_index()
        reset.columns = reset.columns.droplevel(level=1)
        reset['date'] = column[1]
        reset = reset.set_index('date')
        day_dfs.append(reset)
    return pd.concat(day_dfs).dropna().sort_index()


def split_sentence(sentence):
    """Split a sentence into phrases."""
    try:
        text = BeautifulSoup(sentence.text, features='html.parser').get_text()
    except TypeError:
        return []
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    split_text = [word for word in text.split() if len(
        word) > 1 and word not in STOPWORDS]
    phrases = []
    for num in NGRAMS:
        for grams in ngrams(split_text, num):
            if grams:
                phrases.append(' '.join(grams))
    return phrases


def main():
    """Main."""
    if IMPORT_ALL:
        csv_files = sorted(glob.glob(f'{DOWNLOAD_DIR}/hackernews*.csv'))
    else:
        csv_files = [download_hackernews()]
    dfs = []
    for csv_file in csv_files:
        dfs.append(pd.read_csv(
            csv_file, index_col=0, parse_dates=True,
            infer_datetime_format=True))
    hackernews_df = pd.concat(dfs)

    hackernews_df = hackernews_df[hackernews_df['text'] != '<null>']
    hackernews_df['text'] = hackernews_df['text'].str.lower()
    latest_timestamp_str = str(hackernews_df.index.max())
    # Start from last checkpoint
    if not IMPORT_ALL and exists(HACKERNEWS_TIMESTAMP):
        with open(HACKERNEWS_TIMESTAMP, encoding='utf-8') as input_file:
            previous_timestamp = pd.Timestamp(input_file.read())
        hackernews_df = hackernews_df[hackernews_df.index > previous_timestamp]
        if len(hackernews_df) == 0:
            return
    # Remove time component
    hackernews_df.index = pd.DatetimeIndex(
        hackernews_df.index.strftime('%Y-%m-%d'))
    hackernews_df.index.name = 'date'
    warnings.filterwarnings('ignore', category=UserWarning, module='bs4')
    hackernews_series = hackernews_df.apply(split_sentence, axis=1)
    hackernews_df = pd.DataFrame({'gram': hackernews_series}).explode('gram')
    dfs = []
    for idx in sorted(hackernews_df.index.unique()):
        day_df = hackernews_df.loc[idx]
        day_df = pd.DataFrame(
            {'count': day_df.groupby(['gram']).size()}).reset_index()
        day_df = day_df.assign(date=idx).set_index('date')
        dfs.append(day_df)
    hackernews_df = pd.concat(dfs)
    if not IMPORT_ALL and exists(STORAGE_FILE):
        hackernews_df = pd.concat(
            [pd.read_pickle(STORAGE_FILE), hackernews_df])
    hackernews_df = sum_duplicates(hackernews_df)
    # Get rid of single phrases
    hackernews_df = hackernews_df[hackernews_df['count'] > 1]
    hackernews_df.to_pickle(STORAGE_FILE, compression='gzip')
    with open(HACKERNEWS_TIMESTAMP, 'w', encoding='utf-8') as output_file:
        output_file.write(latest_timestamp_str)
    if len(csv_files) == 1:
        subprocess.run(['gzip', csv_files[0]], check=True)


if __name__ == '__main__':
    main()
