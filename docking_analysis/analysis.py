import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

headers = ["Name", "Visible", "Color", "Parent", "Distance", "Category", "Types", "From", "From Chemistry", "To", "To Chemistry",]


def get_DS_pocket_residues(path, chain_id):
    temp_ls = []
    df = pd.read_csv(path, sep='\t', header=None)
    num_cols = df.shape[1]
    full_headers = headers + [f"Extra_{i}" for i in range(len(headers), num_cols)]
    df.columns = full_headers
    df = df[df['Distance'] < 4.5]
    for name in df['Name'].to_list():
        words = name.split('-')
        for word in words:
            word = word.strip()
            if word.startswith(chain_id):
                temp_ls.append(word.split(':')[1])
    return temp_ls



def get_dic(path, id, chain_id):
    df = pd.read_csv(path, sep='\t', header=None)
    num_cols = df.shape[1]
    full_headers = headers + [f"Extra_{i}" for i in range(len(headers), num_cols)]
    df.columns = full_headers
    df = df[df['Distance'] < 4.5]
    d = {k: 0 for k in ls}
    for name in df['Name'].dropna().to_list():
        words = name.split('-')
        for word in words:
            word = word.strip()
            if word.startswith(chain_id):
                res = word.split(':')[1]
                if res in d:
                    d[res] += 1
    d['id'] = id
    return d

import os
ls = []
files = ['DPP4-Diprotin A (1NU8)', 'DPP4-Sitagliptin (1X70)', 'DPP4-YPQPQ (ADCP)', 'DPP4-YPQPQ (AlphaFold3)', 'DPP4-YPQPQ (LibDock)', 'DPP4-YPQPQ (VINA)']
chain_ids = ['B', 'B' , 'A' , 'A', 'A', 'A']
for i,file in enumerate(files):
    print(file, get_DS_pocket_residues(f'ds_fp/{file}.txt', chain_ids[i]))
    ls += get_DS_pocket_residues(f'ds_fp/{file}.txt', chain_ids[i])

ls = list(set(ls))
ls.remove('715801')
all_data = []



for i,file in enumerate(files):
    all_data.append(get_dic(f'ds_fp/{file}.txt', file, chain_ids[i]))
plot_df = pd.DataFrame(all_data)
plot_df = plot_df.set_index('id')

residue_cols = [col for col in plot_df.columns if col != 'id']

def shift_residue(res):
    aa = res[:3]
    try:
        num = int(res[3:])
        return f"{aa}{num}"
    except ValueError:
        return res


new_col_names = {res: shift_residue(res) for res in residue_cols}
plot_df = plot_df.rename(columns=new_col_names)

def extract_number(res):
    return int(res[3:])

sorted_cols = sorted(plot_df.columns, key=extract_number)
plot_df = plot_df[sorted_cols]

plt.figure(figsize=(max(10, len(plot_df.columns) * 0.4), 5))
sns.heatmap(plot_df, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Interaction Count'})
plt.title('Residue Contact Frequency (<4.5 Å)')
plt.xlabel('Residue')
plt.ylabel('Model')
plt.tight_layout()
plt.savefig('heatmap.png', dpi=300)
plt.show()