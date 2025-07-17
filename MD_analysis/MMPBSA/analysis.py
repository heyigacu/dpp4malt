import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap


PARENT_DIR = os.path.abspath(os.path.dirname(__file__))

TITLE_FONT = {'family':'Arial', 'size':16, 'fontweight':'bold'}
LABEL_FONT = {'family':'Arial','size':14,'fontweight':'bold'}
TICKSLABEL_FONT = {'labelfontfamily':'Arial','labelsize':10,'pad':0.03}
LEGEND_FONT = {'family':'Arial', 'size': 12, 'weight': 'bold'}
SS_DIC = {"None":0,"Ext":1,"Bridge":2,"3-10":3,"Alpha":4,"Pi":5,"Turn":6,"Bend":7}
color_list = ['#EC3E31','#4F99C9','#A8D3A0']

def decomp2matrix(decomp_data_path):
    ls_total = []
    subject1 = []
    subject2 = []
    flag = 0
    with open(decomp_data_path,'r') as fr:
        lines = fr.readlines()
        for line in lines:
            if line.startswith('DELTAS'):
                flag = 1
            elif flag == 1 and line.startswith('Residue'):
                words = line.split(',')
                words[-1] = ''
                for i,word in enumerate(words):
                    if word == '':
                        words[i] = words[i-1]
                subject1 = words
                
            elif flag == 1 and line.startswith(',,Avg'):
                words = line.strip().split(',')
                words = words + ['Avg.', 'Std. Dev.', 'Std. Err. of Mean']
                for i,word in enumerate(words):
                    subject2.append(subject1[i]+' '+word)
            elif flag == 1 and line[0].isalpha():
                ls_total.append([float(_) if i > 1 else _.strip() for i,_ in enumerate(line.strip().split(','))])
            # elif line.startswith('Sidechain') or (flag==2 and (line.startswith('Residue') or line.startswith(',,Avg'))):
            #     flag = 2
            # elif flag == 2 and line[0].isalpha() and not line.startswith('Residue'):
            #     ls_side.append([float(_) if i > 1 else _ for i,_ in enumerate(line.strip().split(','))])
            # elif line.startswith('Backbone') or (flag==3 and (line.startswith('Residue') or line.startswith(',,Avg'))):
            #     flag = 3 
            # elif flag == 3 and line[0].isalpha() and not line.startswith('Residue'):
            #     ls_back.append([float(_) if i > 1 else _ for i,_ in enumerate(line.strip().split(','))])
            else:flag=0

    return pd.DataFrame(ls_total,columns=subject2)

def dic2errorbar(dic,subs,delta):
    fig = plt.figure(figsize=(8,5))
    for i,decomp_name in enumerate(['DELTAS']):
        ax = fig.add_subplot(1,1,i+1)
        ls = dic
        for j,sub in enumerate(subs): 
            print(len(ls))
            df = ls[j]
            # for item in ['Internal','van der Waals','Electrostatic','Polar Solvation','Non-Polar Solv.','TOTAL']:
            for item in ['TOTAL']:
                ax.errorbar(df['Residue '], df[item+' '+'Avg.'], yerr=df[item+' '+'Std. Err. of Mean'], capsize=2, capthick=1, linewidth=1.5, barsabove=True, alpha=0.7, label=sub, color=color_list[j])
            ax.set_xticks(ls[j]['Residue '],[''.join([name.split()[0], str((int(name.split()[1])+delta))]) for name in ls[j]['Residue ']],rotation=90,fontweight='bold')
            from matplotlib.font_manager import FontProperties
            font_prop = FontProperties(weight='bold')
            for label in ax.get_yticklabels():
                label.set_fontproperties(font_prop)
            ax.tick_params(**TICKSLABEL_FONT)
            ax.set_xlabel('Residue',fontdict=LABEL_FONT)
            ax.set_ylabel(f'{decomp_name} Energy (kcal/mol)',fontdict=LABEL_FONT)
            ax.invert_yaxis()
            ax.legend(subs,frameon=False,prop=LEGEND_FONT)
    plt.tight_layout()
    plt.savefig('mmpbsa.png',dpi=300)   

residues = [48, 125, 201, 205, 206, 209, 357, 471, 545, 546, 547, 548, 549, 554, 560, 562, 563, 627, 628, 629, 630, 631, 632, 633, 634, 654, 655, 656, 659, 662, 663, 666, 669, 710, 711, 740, 741, 752]


def plt_mmpbsa(subs=['Diprotin_A'], residues=residues, delta=38):
    residues = [i-delta for i in residues]
    ls = [] 
    for sub in subs:
        df_total = decomp2matrix('{}.dat'.format(sub))
        

        # ls.append(df_total.iloc[residues])
        df_total = df_total.iloc[1:725, :]
        top30 = df_total.sort_values(by='TOTAL Avg.', ascending=True).head(30)
        ls.append(top30)


    dic2errorbar(ls, subs, delta)

if __name__=='__main__':
    plt_mmpbsa()
