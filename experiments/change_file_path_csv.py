import sys
sys.path.append('/root/Generative-Models/ReDock/')
import os
import argparse
import pandas as pd
from utils.inference_utils import set_nones
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--protein_ligand_csv_folder", type=str, default="data/crossdock_csv", help="Path to a folder of .csv file specifying the input as described in the README.")
args = parser.parse_args()

for csv_file in tqdm(os.listdir(args.protein_ligand_csv_folder)):
    # delete the str "/home/haotian/Molecule_Generation/Flexible-docking/" in the csv file
    csv_file_path = os.path.join(args.protein_ligand_csv_folder, csv_file)
    df = pd.read_csv(csv_file_path)
    protein_path_list = set_nones(df["protein_path"].tolist())
    ligand_path_list = set_nones(df["ligand_description"].tolist())
    new_protein_path_list = []
    new_ligand_path_list = []

    for protein_path in protein_path_list:
        new_protein_path_list.append(protein_path.replace('/home/haotian/Molecule_Generation/Flexible-docking/', ''))
    for ligand_path in ligand_path_list:
        new_ligand_path_list.append(ligand_path.replace('/home/haotian/Molecule_Generation/Flexible-docking/', ''))

    df['protein_path'] = new_protein_path_list
    df['ligand_description'] = new_ligand_path_list
    # drop the column "ligand_path"
    df.drop(['ligand_path'], axis=1, inplace=True)
    # replace the original csv file
    df.to_csv(csv_file_path, index=False)