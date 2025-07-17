import pandas as pd
from collections import Counter
df = pd.read_csv('KEGG enrichment.csv')

ls = []
for string in list(df['geneID']):
    ls+=string.split('/')

print(Counter(ls))