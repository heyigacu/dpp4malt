from Bio.PDB import PDBParser

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def get_residue_atom_map(pdb_file, delta):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_file)
    
    residue_atom_map = {}

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ':  # 排除水和异构体等
                    continue
                res_id = residue.id[1]   # 残基编号
                name = residue.get_resname()
                atom_ids = [atom.serial_number-1 for atom in residue]
                residue_atom_map[res_id+delta] = atom_ids

    return residue_atom_map


residue_atom_map = get_residue_atom_map('DA/struc.pdb',38)
atom_to_residue = {}
for res_id, atom_ids in residue_atom_map.items():
    for atom_id in atom_ids:
        atom_to_residue[atom_id] = res_id

lig_df = pd.read_csv('DA/lig_donor.csv', sep='\t')
rec_df = pd.read_csv('DA/rec_donor.csv', sep='\t')


df = pd.concat([lig_df, rec_df], ignore_index=True)

df['donor_res'] = df['donor_idx'].map(atom_to_residue)
df['acceptor_res'] = df['acceptor_idx'].map(atom_to_residue)
df['hbond_pair'] = df['donor_res'].astype(str) + ' → ' + df['acceptor_res'].astype(str)


pair_counts = df['hbond_pair'].value_counts(normalize=True).reset_index()
pair_counts.columns = ['hbond_pair', 'frequency']
pair_counts_top20 = pair_counts.head(20)

plt.figure(figsize=(8, 6))
sns.barplot(data=pair_counts_top20, x='frequency', y='hbond_pair', palette="Blues_d")
plt.xlabel("Occurrence Frequency (%)")
plt.ylabel("Donor → Acceptor Residue")
plt.title("Hydrogen Bond Pair Occupancy")
plt.tight_layout()
plt.savefig('Occupancy.png')

plt.figure(figsize=(8, 6))
sns.kdeplot(
    data=df, x="distance", y="angle",
    fill=True, cmap="mako", thresh=0.05
)
plt.xlabel("H-Bond Distance (Å)")
plt.ylabel("H-Bond Angle (°)")
plt.title("Hydrogen Bond Distance vs Angle Density")
plt.tight_layout()
plt.savefig('Stable.png')


frame_counts = df.groupby('frame').size().reset_index(name='hbond_count')
plt.figure(figsize=(8, 6))
sns.lineplot(data=frame_counts, x='frame', y='hbond_count')
plt.xlabel("Frame")
plt.ylabel("Number of Hydrogen Bonds")
plt.title("Number of Hydrogen Bonds over Time")
plt.tight_layout()
plt.savefig('Number.png')