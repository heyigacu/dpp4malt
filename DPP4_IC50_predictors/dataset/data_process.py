import pandas as pd
import rdkit.Chem as Chem
import numpy as np
from sklearn.model_selection import KFold
import os
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
parent_dir = os.path.dirname(os.path.abspath(__file__))


def calculate_rdkit_descriptors(smiles_list):
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    descriptor_data = []
    valid_smiles = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            try:
                descriptor_values = calculator.CalcDescriptors(mol)
                descriptor_data.append(descriptor_values)
                valid_smiles.append(smi)
            except:
                continue

    descriptors_df = pd.DataFrame(descriptor_data, columns=descriptor_names)
    return descriptors_df

def neglog(ic50_values):
    pic50_values = -np.log10(ic50_values * 1e-6)
    return pic50_values

def neglogback(pic50_values):
    ic50_values = 10**-pic50_values * 1e6
    return ic50_values

def drop_rdkit_error(df):
    drop_ids = []
    for index,row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['Smiles'])
            if mol == None:
                drop_ids.append(index)
        except:
            drop_ids.append(index)
    df = df.drop(drop_ids)
    return df

def scaffold_split(df):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(df))
    for i, (train_index, test_index) in enumerate(folds):
        train_df = pd.DataFrame({'index': train_index})
        test_df = pd.DataFrame({'index': test_index})
        train_df.to_csv(parent_dir+f'/splits/fold_{i}_train_indices.csv', index=False)
        test_df.to_csv(parent_dir+f'/splits/fold_{i}_test_indices.csv', index=False)

def read_fold_indices(fold_number):
    train_data = pd.read_csv(parent_dir+f'/splits/fold_{fold_number}_train_indices.csv')
    test_data = pd.read_csv(parent_dir+f'/splits/fold_{fold_number}_test_indices.csv')
    train_index = train_data['index'].tolist()
    test_index = test_data['index'].tolist()
    return train_index, test_index

def process_raw_dpp4_ic50():
    # df = pd.read_csv(parent_dir+'/raw_dpp4_ic50.csv',header=0,sep='\t')
    # df = df[["Smiles","Standard Relation","Standard Value"]]
    
    # df.columns = ['Smiles','Relation','IC50']
    # df = drop_rdkit_error(df)
    # df = df[df['Relation'] =='\'=\'']
    # df.insert(len(df.columns),'pIC50',neglog(df['IC50']))
    
    # df.to_csv(parent_dir+'/clean_dpp4_ic50.csv',index=False,header=True,sep='\t')
    df = pd.read_csv(parent_dir+'/clean_dpp4_ic50.csv',header=0,sep='\t')
    descriptors_df = calculate_rdkit_descriptors(df['Smiles'])
    descriptors_df.to_csv(parent_dir+'/descriptors.csv', sep='\t', index=False)
    scaffold_split(df)

def peptide_to_smiles(seq):
    try:
        mol = Chem.MolFromFASTA(seq)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except:
        return None


# def process_external_test():
#     df = pd.read_csv(parent_dir+'/external_test.csv',header=0,sep='\t')
#     df['Smiles'] = df['Name'].apply(peptide_to_smiles)
#     df['IC50'] = df['ActiveValue'] * 1000
#     df.insert(len(df.columns),'pIC50',neglog(df['IC50']))
#     df.to_csv(parent_dir+'/clean_external_test.csv',index=False,header=True,sep='\t')
#     df = pd.read_csv(parent_dir+'/clean_external_test.csv',header=0,sep='\t')
#     descriptors_df = calculate_rdkit_descriptors(df['Smiles'])
#     descriptors_df.to_csv(parent_dir+'/test_descriptors.csv', sep='\t', index=False)

def process_external_test():
    df = pd.read_csv(parent_dir+'/external_test.csv',header=0,sep='\t')
    df.insert(len(df.columns),'pIC50',neglog(df['IC50']))
    df.to_csv(parent_dir+'/clean_external_test.csv',index=False,header=True,sep='\t')
    df = pd.read_csv(parent_dir+'/clean_external_test.csv',header=0,sep='\t')
    descriptors_df = calculate_rdkit_descriptors(df['Smiles'])
    descriptors_df.to_csv(parent_dir+'/test_descriptors.csv', sep='\t', index=False)



if __name__ == '__main__':
    # process_raw_dpp4_ic50()
    process_external_test()

