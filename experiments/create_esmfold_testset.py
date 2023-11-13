import sys
sys.path.append('/root/Generative-Models/ReDock/')
import os
import argparse
import pandas as pd
from utils.inference_utils import set_nones

parser = argparse.ArgumentParser()
parser.add_argument("--protein_ligand_csv", type=str, default="data/testset_csv.csv", help="Path to a .csv file specifying the input as described in the README.")
parser.add_argument("--new_csv", type=str, default="data/esmfold_testset_csv.csv", help="Path to a .csv file specifying the input as described in the README.")
args = parser.parse_args()

df = pd.read_csv(args.protein_ligand_csv)
complex_name_list = set_nones(df["complex_name"].tolist())
protein_path_list = set_nones(df["protein_path"].tolist())

for i, name in enumerate(complex_name_list):
    if name is None:
        complex_name_list[i] = f'complex_{i}'
    elif isinstance(name, int): # rename number to pdbid
        # get dirname of path, then get the last folder name
        complex_name_list[i] = os.path.split(os.path.dirname(protein_path_list[i]))[1]

new_protein_path_list = []
for protein_path in protein_path_list:
    # new_protein_path_list.append(protein_path.replace('processed', ''))
    file_name = os.path.basename(protein_path)
    new_file_name = file_name.replace('processed', 'esmfold_aligned_tr')
    new_protein_path_list.append(protein_path.replace(file_name, new_file_name))

df['protein_path'] = new_protein_path_list
df['complex_name'] = complex_name_list

df.to_csv(args.new_csv, index=False)

