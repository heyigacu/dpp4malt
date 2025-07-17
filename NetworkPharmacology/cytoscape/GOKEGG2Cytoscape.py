import pandas as pd
import os
parent_dir = os.path.abspath(os.path.dirname(__file__))



def process_KEGG():
    df = pd.read_csv(parent_dir+f'/pathways', sep=',')
    print(df.columns)
    ls_gene = []
    ls_des = []
    with open(parent_dir+f'/KEGG_edge.csv','w') as f:
        f.write('Source,Target\n')
        for index,row in df.iterrows():
            des = row['Description']
            des = row['pathway']
            ls_des.append(des)
            genes = row['geneID'].strip().split('/')
            ls_gene+=genes
            for gene in genes:
                f.write(f'{gene},{des}\n')

    with open(parent_dir+f'/KEGG_node.csv','w') as f:
        f.write('Node,Type\n')
        for gene in list(set(ls_gene)):
            f.write(f'{gene}\tGene\n')
        for des in list(set(ls_des)):
            f.write(f'{des}\tDisease\n')

process_KEGG()
