import os
import pandas as pd
import os
import argparse
from numpy import ceil
import multiprocessing as mp

DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='vina docking')
parser.add_argument("-t", "--taskname", type=str, default='herbcnki',
                    help="mol2 input file")
parser.add_argument("-v", "--vina", type=str, default='/home/hy/Softwares/vina/autodock_vina_1_1_2_linux_x86/bin/vina',
                    help="vina path")
parser.add_argument("-b", "--box", nargs='+', help='<Required> Set flag', default='40.9 50.5 38.5 25.4 16.8 22.0')
parser.add_argument("-r", "--receptor", type=str, default='dpp4.pdbqt',
                    help="receptor file")
parser.add_argument("-n", "--num_cores", type=int, default=24,
                    help="receptor file")
args = parser.parse_args()

taskname = args.taskname
inputfile = os.path.join(DIR,f'{taskname}_pdbqt.csv')
outfile = os.path.join(DIR,f"{taskname}_dockout.csv")
ligand_pdbqt_dir = os.path.join(DIR,f'{taskname}_pdbqts')
args.box = args.box.split()
center_x = float(args.box[0])
center_y = float(args.box[1])
center_z = float(args.box[2])
size_x = float(args.box[3])
size_y = float(args.box[4])
size_z = float(args.box[5])
vina = args.vina
receptor = args.receptor
num_cores = args.num_cores

dock_save_dir =os.path.join(DIR,f'{taskname}_docks')
if not os.path.exists(dock_save_dir):
    os.mkdir(dock_save_dir)
dock_log_dir =os.path.join(DIR,f'{taskname}_dock_logs')
if not os.path.exists(dock_log_dir):
    os.mkdir(dock_log_dir)


def vina_dock(taskname, df):
    dock_paths = []
    
    for index,row in df.iterrows():
        pdbqt_path = row['pdbqt_path']
        if pdbqt_path != 'error' and os.path.exists(pdbqt_path):
            try:
                dock_out_path = '{}/dock_out{}.pdbqt'.format(dock_save_dir, index)
                log_path = '{}/dock_out{}.txt'.format(dock_log_dir, index)
                os.system(f'{vina} --center_x {center_x} --center_y {center_y} --center_z {center_z} --size_x {size_x} --size_y {size_y} --size_z {size_z} \
                        --receptor {receptor} --ligand {pdbqt_path} --out {dock_out_path} --log {log_path} --exhaustiveness 8')
                dock_paths.append(dock_out_path)
            except:
                dock_paths.append('error')
        else:
            dock_paths.append('error')
    return dock_paths       

def write_dock_log(df):
    dock_out_ls = []
    affnity_ls = []
    with open(f'{taskname}_vina_dock.log', 'w') as log:
        for index,row in df.iterrows():
            log_path = '{}/dock_out{}.txt'.format(dock_log_dir, index)
            dock_out_path = '{}/dock_out{}.pdbqt'.format(dock_save_dir, index)
            if os.path.exists(dock_out_path):
                dock_out_ls.append(dock_out_path)
                affnity_ls.append(open(dock_out_path,'r').readlines()[1].split()[3])
            else:
                dock_out_ls.append('error')
                affnity_ls.append(99999.)     
            log.write('>mol{}\n'.format(index))
            if os.path.exists(log_path):
                with open(log_path,'r') as log_temp:
                    lines = log_temp.readlines()
                    for line in lines:
                        log.write(line)
            else:
                log.write('error')
    return dock_out_ls, affnity_ls



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

    results = [pool.apply_async(vina_dock, args=(name, param)) for name, param in dic.items()]
    results = [p.get() for p in results]
    return results
    


df = pd.read_csv(inputfile,sep='\t',header=0)
# vina_dock(taskname, df)
multi_tasks(df)
dock_out_ls, affnity_ls = write_dock_log(df)
df.insert(df.shape[1],'dock_out_path',dock_out_ls)
df.insert(df.shape[1],'affinity',affnity_ls)
df.to_csv(outfile,header=True,index=False,sep='\t')

