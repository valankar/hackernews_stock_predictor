Steampipe

(hackernews + reddit + twitter) -> AI -> stock predictions

Look for 1, 2, 3 word phrases in top stories and see if correlation to stock price changes.

"train"
Look at words from 2 days ago, then stock price changes 1 day ago.

Any site that lets you play with money and investing?

```shell
steampipe query "select time, text  from hackernews_item where type='comment' order by time asc" --output csv
```

```python

# HashingVectorizer
new_df, _ = load_data()
# new_df = hackernews_df.reset_index()
# Explode rows by counts
# new_df = new_df.loc[new_df.index.repeat(new_df['count'].astype(int))].drop(columns='count').set_index('date')

hv = HashingVectorizer(n_features=10)
vector = hv.transform(new_df['gram'])
sdf = pd.DataFrame.sparse.from_spmatrix(vector, index=new_df.index)
sdf = sdf.multiply(new_df['count'], axis='index')
sdf = sdf.groupby(sdf.index).sum()
```