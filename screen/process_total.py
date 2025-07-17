import pandas as pd
import numpy as np
df = pd.read_csv('maiya_data.csv',sep='\t',header=0)

def neglogback(pic50_values):
    ic50_values = 10**-pic50_values * 1e6
    return ic50_values


df['KPGT_pIC50'] = pd.to_numeric(df['KPGT_pIC50'], errors='coerce')
df['affinity'] = df['affinity'].replace(99999, np.nan)
df['Libdock'] = pd.to_numeric(df['Libdock'], errors='coerce')
mean_KPGT_pIC50 = df['KPGT_pIC50'].mean()
mean_affinity = df['affinity'].mean()
mean_Libdock = df['Libdock'].mean()
df['KPGT_pIC50'].fillna(mean_KPGT_pIC50, inplace=True)
df['affinity'].fillna(mean_affinity, inplace=True)
df['Libdock'].fillna(mean_Libdock, inplace=True)
df['KPGT_IC50'] = df['KPGT_pIC50'].apply(neglogback)

print(df.columns)

# 'Error Format'
# 99999
# 'error'

def calculate_rankings(df):
    df['affinity_rank'] = df['affinity'].rank(method='min', ascending=True)
    df['KPGT_IC50_rank'] = df['KPGT_IC50'].rank(method='min', ascending=True)
    df['Conplex_rank'] = df['Conplex'].rank(method='min', ascending=False)
    df['Libdock_rank'] = df['Libdock'].rank(method='min', ascending=False)
    df['average_rank'] = df[['affinity_rank', 'KPGT_IC50_rank', 'Conplex_rank', 'Libdock_rank']].mean(axis=1)
    return df

df_with_rankings = calculate_rankings(df)
df_with_rankings.to_csv('total_rank.csv',sep='\t',index=False)
