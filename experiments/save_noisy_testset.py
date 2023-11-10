# Copy test set holo structure and holo structure with noise
import sys
sys.path.append('/root/Generative-Models/ReDock/')
import copy
import os

import numpy as np
from argparse import ArgumentParser, Namespace
from torch_geometric.loader import DataLoader
from datasets.pdbbind import PDBBind
from utils.inference_utils import set_nones
from utils.sampling import randomize_position
from utils.visualise import ModifiedPDB
from tqdm import tqdm
import yaml
import pandas as pd

parser = ArgumentParser()
parser.add_argument('--protein_ligand_csv', type=str, default='data/testset_csv.csv', help='Path to a .csv file specifying the input as described in the README. If this is not None, it will be used instead of the --protein_path, --protein_sequence and --ligand parameters')
parser.add_argument('--complex_name', type=str, default='1a0q', help='Name that the complex will be saved with')
parser.add_argument('--protein_path', type=str, default=None, help='Path to the protein file')
parser.add_argument('--protein_sequence', type=str, default=None, help='Sequence of the protein for ESMFold, this is ignored if --protein_path is not None')
parser.add_argument('--ligand_description', type=str, default='CCCCC(NC(=O)CCC(=O)O)P(=O)(O)OC1=CC=CC=C1', help='Either a SMILES string or the path to a molecule file that rdkit can read')
parser.add_argument('--esm_embeddings_path', type=str, default='data/esm2_output', help='Path to folder where the ESM embeddings are stored')

parser.add_argument('--out_dir', type=str, default='results/user_holo_testset', help='Directory where the outputs will be written to')
parser.add_argument('--cache_path', type=str, default='data/example/dataset_cache', help='Path to folder where the cache is stored')
parser.add_argument('--model_dir', type=str, default='workdir/ReDock_baseline', help='Path to folder with trained score model and hyperparameters')
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for preprocessing')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
with open(f'{args.model_dir}/model_parameters.yml') as f:
    score_model_args = Namespace(**yaml.full_load(f))
if args.protein_ligand_csv is not None:
    df = pd.read_csv(args.protein_ligand_csv)
    complex_name_list = set_nones(df['complex_name'].tolist())
    protein_path_list = set_nones(df['protein_path'].tolist())
    protein_sequence_list = set_nones(df['protein_sequence'].tolist())
    ligand_description_list = set_nones(df['ligand_description'].tolist())
else:
    complex_name_list = [args.complex_name]
    protein_path_list = [args.protein_path]
    protein_sequence_list = [args.protein_sequence]
    ligand_description_list = [args.ligand_description]

for i, name in enumerate(complex_name_list):
    if name is None:
        complex_name_list[i] = f'complex_{i}'
    elif isinstance(name, int): # rename number to pdbid
        # get dirname of path, then get the last folder name
        complex_name_list[i] = os.path.split(os.path.dirname(protein_path_list[i]))[1]

for name in complex_name_list:
    write_dir = f'{args.out_dir}/{name}'
    os.makedirs(write_dir, exist_ok=True)

test_dataset = PDBBind(transform=None, root='',
                       protein_path_list=protein_path_list,
                       ligand_descriptions=ligand_description_list, 
                       limit_complexes=0,
                       receptor_radius=score_model_args.receptor_radius,
                       cache_path=args.cache_path,
                       remove_hs=score_model_args.remove_hs, max_lig_size=None,
                       c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
                       matching=False, keep_original=False,
                       popsize=score_model_args.matching_popsize,
                       maxiter=score_model_args.matching_maxiter,
                       all_atoms=score_model_args.all_atoms,
                       atom_radius=score_model_args.atom_radius,
                       atom_max_neighbors=score_model_args.atom_max_neighbors,
                       esm_embeddings_path=args.esm_embeddings_path,
                       require_ligand=True,
                       num_workers=args.num_workers)

test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
for idx, orig_complex_graph in tqdm(enumerate(test_loader), desc="Generating noisy holo Conformation", total=len(test_dataset)):
    # copy original complex files
    protein_path = orig_complex_graph["name"][0].split('____')[0]
    ligand_description = orig_complex_graph["name"][0].split('____')[-1]
    complex_name = os.path.split(os.path.dirname(protein_path))[1]
    write_dir = f'{args.out_dir}/{complex_name}'

    # copy protein file using os 
    os.system(f'cp {protein_path} {write_dir}/')
    # copy ligand file using os
    os.system(f'cp {ligand_description} {write_dir}/')

    # save noisy holo structure
    data_list = [copy.deepcopy(orig_complex_graph)]
    randomize_position(data_list, score_model_args.no_torsion, False, score_model_args.no_chi_angle, 
                       score_model_args.tr_sigma_max, score_model_args.atom_radius, score_model_args.atom_max_neighbors)
    protein_atom_pos = np.asarray([complex_graph['atom'].pos.numpy() + orig_complex_graph.original_center.numpy() for complex_graph in data_list])

    for index, pos in enumerate(protein_atom_pos):
        mod_prot = mod_prot = ModifiedPDB(pdb_path=protein_path, 
                                          ligand_description=ligand_description, pocket_pos=pos)
        mod_prot.to_pdb(os.path.join(write_dir, f'{complex_name}_noisy_holo_{index}.pdb'))




