"""Improved generation script for large scale realistic scenerios
currently supports crossdocking, redocking, apodock, and virtual_screen
"""
import copy
import os
import torch
from argparse import ArgumentParser, Namespace
from rdkit.Chem import RemoveHs
from functools import partial
import numpy as np
import pandas as pd
from rdkit import RDLogger
from torch_geometric.loader import DataLoader, DataListLoader

from datasets.process_mols import write_mol_with_coords
from datasets.inference_datasets import InferenceDatasets
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.inference_utils import set_nones
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.visualise import PDBFile, ModifiedPDB
from tqdm import tqdm
from utils.so2 import SO2VESchedule
import pickle
import pdb

RDLogger.DisableLog('rdApp.*')
import yaml
parser = ArgumentParser()
# Inference mode
parser.add_argument('--mode', type=str, default='virtual_screen', help='Inferece mode, [crossdock, redocking, apodock. virtual_screen]')
parser.add_argument('--cache_path', type=str, default=None, help='Path to folder where the cache are stored')
# For interface for various settings except virtual_screen
parser.add_argument('--protein_ligand_csv', type=str, default=None, help='Path to a .csv file specifying the input as described in the README. If this is not None, it will be used instead of the --protein_path, --protein_sequence and --ligand parameters')
# demo or one sample only
parser.add_argument('--protein_path', type=str, default=None, help='Path to the protein file')
parser.add_argument('--protein_sequence', type=str, default=None, help='Sequence of the protein for ESMFold, this is ignored if --protein_path is not None')
parser.add_argument('--ligand_description', type=str, default='CCCCC(NC(=O)CCC(=O)O)P(=O)(O)OC1=CC=CC=C1', help='Either a SMILES string or the path to a molecule file that rdkit can read')
parser.add_argument('--complex_name', type=str, default='1a0q', help='Name that the complex will be saved with')
# For virtual_screen
parser.add_argument('--protein_target_path', type=str, default=None, help='Path to the target protein file')
parser.add_argument('--ligand_in_pocket_path', type=str, default=None, help='Path to the ligand in that defines the pocket')
parser.add_argument('--ligand_database_path', type=str, default=None, help='Path to the ligand database file (usually sdf) for screening')
parser.add_argument('--start_ligand_id', type=int, default=1, help='The ligand id in the database to start virtual screening from, 1 is the first, not 0')
parser.add_argument('--end_ligand_id', type=int, default=-1, help='The ligand id in the database to end virtual screening at')
# save the results
parser.add_argument('--out_dir', type=str, default='results/user_inference', help='Directory where the outputs will be written to')
# model configurations
parser.add_argument('--model_dir', type=str, default='workdir/paper_score_model', help='Path to folder with trained score model and hyperparameters')
parser.add_argument('--ckpt', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use for the score model')
parser.add_argument('--confidence_model_dir', type=str, default='workdir/paper_confidence_model', help='Path to folder with trained confidence model and hyperparameters')
parser.add_argument('--confidence_ckpt', type=str, default='best_model_epoch75.pt', help='Checkpoint to use for the confidence model')
# sampling configurations
parser.add_argument('--samples_per_complex', type=int, default=5, help='Number of samples to generate')
parser.add_argument('--complex_per_batch', type=int, default=12, help='Number of complexes to generate in parallel') #NOTE: sample_num in one shot = sample_per_complex * complex_per_batch
parser.add_argument('--batch_size', type=int, default=32, help='Number of samples to run in parallel')
parser.add_argument('--no_final_step_noise', action='store_true', default=False, help='Use no noise in the final step of the reverse diffusion')
parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
parser.add_argument('--actual_steps', type=int, default=None, help='Number of denoising steps that are actually performed')
# device configurations
parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for preprocessing')
parser.add_argument('--device', type=str, default="cuda:0", help='Device to run inference on')

# Specicalized Sampling Options
parser.add_argument('--no_chi_angle', action='store_true', default=False, help='Do not sample sidechain chi angles')
parser.add_argument('--no_chi_noise', action='store_true', default=False, help='Do not add noise to sidechain comformations')
args = parser.parse_args()
# print(args)
# pdb.set_trace()

os.makedirs(args.out_dir, exist_ok=True)
with open(f'{args.model_dir}/model_parameters.yml') as f:
    score_model_args = Namespace(**yaml.full_load(f))
if args.confidence_model_dir is not None:
    with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
        confidence_args = Namespace(**yaml.full_load(f))

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# Load the input data, initialize the variable
complex_name_list, protein_path_list, protein_sequence_list, ligand_description_list = None, None, None, None
protein_target_path, ligand_in_pocket_path, ligand_database_path = None, None, None

if args.protein_ligand_csv is not None: # crossdocking or redocking
    df = pd.read_csv(args.protein_ligand_csv)
    complex_name_list = set_nones(df['complex_name'].tolist())
    protein_path_list = set_nones(df['protein_path'].tolist())
    protein_sequence_list = set_nones(df['protein_sequence'].tolist())
    ligand_description_list = set_nones(df['ligand_description'].tolist())
elif args.ligand_database_path is not None: # virtual_screen, it uses a different interface for user's ease
    protein_target_path = args.protein_target_path #TODO: remember to unify the dataset and dataloaer for different modes, so they can share the same interface codes
    ligand_in_pocket_path = args.ligand_in_pocket_path
    ligand_database_path = args.ligand_database_path
    start_ligand_id = args.start_ligand_id
    end_ligand_id = args.end_ligand_id
else: # demo or one sample only
    complex_name_list = [args.complex_name]
    protein_path_list = [args.protein_path]
    protein_sequence_list = [args.protein_sequence]
    ligand_description_list = [args.ligand_description]

# complex_name_list = [name if name is not None else f"complex_{i}" for i, name in enumerate(complex_name_list)]
if args.mode != 'virtual_screen':
    for i, name in enumerate(complex_name_list):
        if name is None:
            complex_name_list[i] = f'complex_{i}'
        elif isinstance(name, int): # rename number to pdbid
            # get dirname of path, then get the last folder name
            complex_name_list[i] = os.path.split(os.path.dirname(protein_path_list[i]))[1]
    # pdb.set_trace()
    # holo_ligand_path = {}
    for name in complex_name_list:
        write_dir = f'{args.out_dir}/{name}'
        os.makedirs(write_dir, exist_ok=True)
        
# pdb.set_trace()
# preprocessing of complexes into geometric graphs
test_dataset = InferenceDatasets(mode=args.mode, cache_dir=args.cache_path, complex_names=complex_name_list,
                                 protein_files=protein_path_list, ligand_descriptions=ligand_description_list, 
                                 protein_sequences=protein_sequence_list, start_ligand_id=start_ligand_id,
                                 protein_target_path=protein_target_path, ligand_in_pocket_path=ligand_in_pocket_path,
                                 ligand_database_path=ligand_database_path, end_ligand_id=args.end_ligand_id,
                                 receptor_radius=score_model_args.receptor_radius,
                                 remove_hs=score_model_args.remove_hs, out_dir=args.out_dir,
                                 c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
                                 all_atoms=score_model_args.all_atoms, atom_radius=score_model_args.atom_radius,
                                 atom_max_neighbors=score_model_args.atom_max_neighbors, num_workers=args.num_workers)

test_loader = DataListLoader(test_dataset, batch_size=args.complex_per_batch, shuffle=False)

t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)
so2_1pi_periodic = SO2VESchedule(pi_periodic=True, cache_folder=score_model_args.diffusion_cache_folder, 
                                    sigma_min=score_model_args.chi_sigma_min, sigma_max=score_model_args.chi_sigma_max, 
                                    annealed_temp=score_model_args.chi_annealed_temp, mode=score_model_args.chi_mode)
so2_2pi_periodic = SO2VESchedule(pi_periodic=False, cache_folder=score_model_args.diffusion_cache_folder, 
                                    sigma_min=score_model_args.chi_sigma_min, sigma_max=score_model_args.chi_sigma_max, 
                                    annealed_temp=score_model_args.chi_annealed_temp, mode=score_model_args.chi_mode)
so2_periodic = [so2_1pi_periodic, so2_2pi_periodic]
model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, so2_periodic=so2_periodic)
state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=True)
model = model.to(device)
model.eval()

t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)
if not args.no_chi_angle:
    so2_1pi_periodic = SO2VESchedule(pi_periodic=True, cache_folder=score_model_args.diffusion_cache_folder, 
                                        sigma_min=score_model_args.chi_sigma_min, sigma_max=score_model_args.chi_sigma_max, 
                                        annealed_temp=score_model_args.chi_annealed_temp, mode=score_model_args.chi_mode)
    so2_2pi_periodic = SO2VESchedule(pi_periodic=False, cache_folder=score_model_args.diffusion_cache_folder, 
                                        sigma_min=score_model_args.chi_sigma_min, sigma_max=score_model_args.chi_sigma_max, 
                                        annealed_temp=score_model_args.chi_annealed_temp, mode=score_model_args.chi_mode)
    so2_periodic = [so2_1pi_periodic, so2_2pi_periodic]
else:
    so2_periodic = None
model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, so2_periodic=so2_periodic)
state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=True)
model = model.to(device)
model.eval()

if args.confidence_model_dir is not None:
    confidence_model = get_model(confidence_args, device, t_to_sigma=t_to_sigma, no_parallel=True, confidence_mode=True, so2_periodic=None, no_chi_angle=args.no_chi_angle)
    state_dict = torch.load(f'{args.confidence_model_dir}/{args.confidence_ckpt}', map_location=torch.device('cpu'))
    confidence_model.load_state_dict(state_dict, strict=True)
    confidence_model = confidence_model.to(device)
    confidence_model.eval()
else:
    confidence_model = None
    confidence_args = None

tr_schedule = get_t_schedule(inference_steps=args.inference_steps)

failures, skipped = 0, 0
N = args.samples_per_complex
print('Size of test dataset: ', len(test_dataset))
for idx, orig_complex_graphs in tqdm(enumerate(test_loader), desc="Generating Docking Conformation", total=len(test_loader)): 
    try:
        confidence_data_list = None
        data_list = []
        orig_list = []
        for orig_complex_graph in orig_complex_graphs: # There could be empty if all original conformers failed (since it may be a failed protein target)
            if not orig_complex_graph.success:
                skipped += 1
                continue
            orig_list.append(orig_complex_graph) # save the original complex graph
            conf_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
            data_list.extend(conf_list)
            
        randomize_position(data_list, score_model_args.no_torsion, False, args.no_chi_noise,
                            score_model_args.tr_sigma_max, score_model_args.atom_radius, score_model_args.atom_max_neighbors)
        # run reverse diffusion
        data_list, confidence = sampling(data_list=data_list, model=model,
                                            inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                                            tr_schedule=tr_schedule, rot_schedule=tr_schedule, tor_schedule=tr_schedule, chi_schedule=tr_schedule,
                                            device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
                                            confidence_model=confidence_model, confidence_data_list=confidence_data_list, 
                                            confidence_model_args=confidence_args, batch_size=args.batch_size, 
                                            no_final_step_noise=args.no_final_step_noise, no_chi_angle=args.no_chi_angle)
        
        for index, orig_complex_graph in enumerate(orig_list):
            # split the ligand_pos into individual complexes
            cur_data_list = data_list[index*N:(index+1)*N]
            cur_confidence = confidence[index*N:(index+1)*N] if confidence is not None else None
            success_status = [complex_graph.success.cpu().numpy() for complex_graph in cur_data_list]
            if not any(success_status):
                failures += 1 # all conformers failed, then this complex failed
            ligand_pos_list = [complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for complex_graph in cur_data_list]
            ligand_pos = np.asarray(ligand_pos_list)
            if not args.no_chi_angle and args.mode != 'virtual_screen':
                protein_atom_pos_list = [complex_graph['atom'].pos.cpu().numpy() + complex_graph.original_center.cpu().numpy() for complex_graph in cur_data_list]
                protein_atom_pos = np.asarray(protein_atom_pos_list)
            # reorder predictions based on confidence output
            if cur_confidence is not None and isinstance(confidence_args.rmsd_classification_cutoff, list):
                cur_confidence = cur_confidence[:, 0]
            if cur_confidence is not None:
                cur_confidence = cur_confidence.cpu().numpy()
                re_order = np.argsort(cur_confidence)[::-1]
                cur_confidence = cur_confidence[re_order]
                success_status = np.asarray(success_status)[re_order]
                ligand_pos = ligand_pos[re_order]
                if not args.no_chi_angle:
                    protein_atom_pos = protein_atom_pos[re_order]

            # save predictions
            if args.mode != 'virtual_screen': # we don't need to save the protein and pocket for virtual_screen
                protein_path = orig_complex_graph["prot_path"]
                ligand_description = orig_complex_graph["lig_path"]
            complex_name = orig_complex_graph["name"]
            # pdb.set_trace()
            lig = orig_complex_graph.mol
            pocket = orig_complex_graph['sidechain']
            # restore the original pocket center
            pocket.node_position = pocket.node_position + orig_complex_graph.original_center
            write_dir = f'{args.out_dir}/{complex_name}'
            if not os.path.exists(write_dir):
                os.makedirs(write_dir) # for virtual_screening, we don't know the complex name before preprocessing
            for rank, pos in enumerate(ligand_pos):
                mol_pred = copy.deepcopy(lig)
                if score_model_args.remove_hs: mol_pred = RemoveHs(mol_pred)
                # add postfix to the complex name according to the success status
                postfix = '' if success_status[rank] else '_rescued'
                if rank == 0: write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}'+postfix+'.sdf'))
                write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}_confidence{cur_confidence[rank]:.2f}'+postfix+'.sdf'))

            if not args.no_chi_angle and args.mode != 'virtual_screen':
                if args.mode == 'apo_docking':
                    pickle.dump(pocket, open(os.path.join(write_dir, f'pocket.pkl'), 'wb')) # save the true pocket object
                for rank, pos in enumerate(protein_atom_pos):
                    # read protein_path and ligand_description from the complex_name, to prevent the error of index shift when preprocessing in PDBBind class failded in some cases
                    # for crossdock, we need to use the holo structure-binding ligand
                    if args.mode != 'redocking':
                    # find the corresponding ligand in holo structure according to the protein name
                        ligand_description = os.path.join(os.path.dirname(protein_path), complex_name.split('_')[0]+'_LIG.sdf') # complex_name[0] is the protein name
                    mod_prot = ModifiedPDB(pdb_path=protein_path, ligand_description=ligand_description, pocket_pos=pos)
                    if args.mode == 'apo_docking':
                        if rank == 0:
                            pickle.dump(pos, open(os.path.join(write_dir, f'rank{rank+1}_pocket_coords.pkl'), 'wb')) # save the predicted pocket comformation
                            mod_prot.to_pdb(os.path.join(write_dir, f'rank{rank+1}_protein.pdb')) # save the predicted protein comformation(with pocket sidechain modification)
                        else:
                            pickle.dump(pos, open(os.path.join(write_dir, f'rank{rank+1}_pocket_coords_confidence{cur_confidence[rank]:.2f}.pkl'), 'wb')) # save the predicted pocket comformation
                            mod_prot.to_pdb(os.path.join(write_dir, f'rank{rank+1}_protein_confidence{cur_confidence[rank]:.2f}.pdb')) # save the predicted protein comformation(with pocket sidechain modification)
                    else:
                        if rank == 0: mod_prot.to_pdb(os.path.join(write_dir, f'rank{rank+1}_protein.pdb'), pocket_only=True)
                        else: mod_prot.to_pdb(os.path.join(write_dir, f'rank{rank+1}_protein.pdb'), pocket_only=True)
    # pdb.set_trace()
    except Exception as e:
        print("Failed on", e)
        if 'non-empty' not in str(e): # ignore the error of empty complex(skipped)
            failures += args.complex_per_batch

print(f'Failed for {failures} complexes')
print(f'Skipped {skipped} complexes')
print(f'Results are in {args.out_dir}')