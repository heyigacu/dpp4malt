import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import os
import argparse
import subprocess
from numpy import ceil
import multiprocessing as mp



DIR = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description='mol2 to pdbqt for ligand')
parser.add_argument("-t", "--taskname", type=str, default='herbcnki',
                    help="mol2 input file")
parser.add_argument("-c", "--cores", type=str, default=24,
                    help="number of cores to use")
args = parser.parse_args()
taskname = args.taskname
inputfile = os.path.join(DIR,f'{taskname}.csv')
outfile = os.path.join(DIR,f"{taskname}_mol2.csv")
num_cores=args.cores
mol2_save_dir = os.path.join(DIR,f'{taskname}_mol2s')
mol2_log_save_dir = os.path.join(DIR,f'{taskname}_mol2_logs')
if not os.path.exists(mol2_save_dir):
    os.mkdir(mol2_save_dir)
if not os.path.exists(mol2_log_save_dir):
    os.mkdir(mol2_log_save_dir)





def generate_mol2(df, taskname):
    ls =[]
    for index,row in df.iterrows():
        lig_path = '{}/ligand{}.mol2'.format(mol2_save_dir, index)
        log_path = '{}/ligand{}.log'.format(mol2_log_save_dir, index)
        mol = Chem.MolFromSmiles(row['Smiles'])
        if mol.GetNumHeavyAtoms() > 50 or mol == None:
            ls.append('error')
            continue
        os.system('obabel -:"{}" -omol2 -O {} --gen3d > {} 2>&1'.format(row['Smiles'],lig_path,log_path))
        print(index)
        if os.path.exists(lig_path):
            ls.append(lig_path)
        else:
            ls.append('error')
    return ls


def write_mol2_log(df):
    ls = []
    with open(os.path.join(DIR,'generate_mol2.log'), 'w') as log:
        for index,row in df.iterrows():
            lig_path = '{}/ligand{}.mol2'.format(mol2_save_dir, index)
            log_path = '{}/ligand{}.log'.format(mol2_log_save_dir, index)
            log.write('convert ligand'+str(index))
            with open(log_path,'r') as log_temp:
                lines = log_temp.readlines()
                for line in lines:
                    log.write(line)
                if lines[0] == '1 molecule converted\n' and os.path.exists(log_path):
                    ls.append(lig_path)
                else:
                    ls.append('error')
    return ls


def check_exits(df):
    ls = []
    for index,row in df.iterrows():
        lig_path = '{}/ligand{}.mol2'.format(mol2_save_dir, index)
        if os.path.exists(lig_path):
            ls.append(lig_path)
        else:
            ls.append('error')
    return ls

def multi_tasks(df):
    num = len(df)
    pool = mp.Pool(num_cores)
    inteval = int(ceil(num/num_cores))
    ls = list(range(0, num, inteval))
    dic={}
    for i in range(len(ls)):
        if i!=(len(ls)-1):
            dic['task'+str(i)] = df[ls[i]:ls[i+1]]
        else:
            dic['task'+str(i)] = df[ls[i]:]

    results = [pool.apply_async(generate_mol2, args=(param, name)) for name, param in dic.items()]
    results = [p.get() for p in results]
    return results
    
df = pd.read_csv(inputfile,header=0,sep='\t')
# mol2_ls = multi_tasks(df)
# mol2_ls = generate_mol2(df,taskname)
# mol2_ls = write_mol2_log(df)
mol2_ls = check_exits(df)
df.insert(df.shape[1],'mol2_path',mol2_ls)
df.to_csv(outfile,sep='\t',index=False,header=True)
