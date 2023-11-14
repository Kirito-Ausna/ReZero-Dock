import copy
import os
import torch
from argparse import ArgumentParser, Namespace
from rdkit.Chem import RemoveHs
from functools import partial
import numpy as np
import pandas as pd
from rdkit import RDLogger
from torch_geometric.loader import DataLoader

from datasets.process_mols import write_mol_with_coords
from datasets.pdbbind import PDBBind
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
parser.add_argument('--protein_ligand_csv', type=str, default=None, help='Path to a .csv file specifying the input as described in the README. If this is not None, it will be used instead of the --protein_path, --protein_sequence and --ligand parameters')
parser.add_argument('--complex_name', type=str, default='1a0q', help='Name that the complex will be saved with')
parser.add_argument('--protein_path', type=str, default=None, help='Path to the protein file')
parser.add_argument('--protein_sequence', type=str, default=None, help='Sequence of the protein for ESMFold, this is ignored if --protein_path is not None')
parser.add_argument('--ligand_description', type=str, default='CCCCC(NC(=O)CCC(=O)O)P(=O)(O)OC1=CC=CC=C1', help='Either a SMILES string or the path to a molecule file that rdkit can read')
parser.add_argument('--esm_embeddings_path', type=str, default='data/embeddings_output', help='Path to folder where the ESM embeddings are stored')

parser.add_argument('--out_dir', type=str, default='results/user_inference', help='Directory where the outputs will be written to')
parser.add_argument('--save_visualisation', action='store_true', default=False, help='Save a pdb file with all of the steps of the reverse diffusion')
parser.add_argument('--samples_per_complex', type=int, default=10, help='Number of samples to generate')

parser.add_argument('--model_dir', type=str, default='workdir/paper_score_model', help='Path to folder with trained score model and hyperparameters')
parser.add_argument('--ckpt', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use for the score model')
parser.add_argument('--confidence_model_dir', type=str, default='workdir/paper_confidence_model', help='Path to folder with trained confidence model and hyperparameters')
parser.add_argument('--confidence_ckpt', type=str, default='best_model_epoch75.pt', help='Checkpoint to use for the confidence model')

parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--no_final_step_noise', action='store_true', default=False, help='Use no noise in the final step of the reverse diffusion')
parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
parser.add_argument('--actual_steps', type=int, default=None, help='Number of denoising steps that are actually performed')

parser.add_argument('--cache_path', type=str, default='data/example/dataset_cache', help='Path to folder where the cache is stored')
parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for preprocessing')

# Specicalized Sampling Options
parser.add_argument('--no_chi_angle', action='store_true', default=False, help='Do not sample sidechain chi angles')
parser.add_argument('--no_chi_noise', action='store_true', default=False, help='Do not add noise to sidechain comformations')
parser.add_argument('--apo_structure', action='store_true', default=False, help='Use apo structure instead of holo structure')
parser.add_argument('--init_pocket_center', action='store_true', default=False, help='Use the center of the pocket as the initial position of the ligand')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
with open(f'{args.model_dir}/model_parameters.yml') as f:
    score_model_args = Namespace(**yaml.full_load(f))
if args.confidence_model_dir is not None:
    with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
        confidence_args = Namespace(**yaml.full_load(f))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# complex_name_list = [name if name is not None else f"complex_{i}" for i, name in enumerate(complex_name_list)]
for i, name in enumerate(complex_name_list):
    if name is None:
        complex_name_list[i] = f'complex_{i}'
    elif isinstance(name, int): # rename number to pdbid
        # get dirname of path, then get the last folder name
        complex_name_list[i] = os.path.split(os.path.dirname(protein_path_list[i]))[1]
# pdb.set_trace()
for name in complex_name_list:
    write_dir = f'{args.out_dir}/{name}'
    os.makedirs(write_dir, exist_ok=True)

# preprocessing of complexes into geometric graphs
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

if args.confidence_model_dir is not None and not confidence_args.use_original_model_cache and args.no_chi_angle:
    print('HAPPENING | confidence model uses different type of graphs than the score model. '
          'Loading (or creating if not existing) the data for the confidence model now.')
    confidence_test_dataset = \
       PDBBind(transform=None, root='', limit_complexes=0,
                               protein_path_list=protein_path_list,
                               ligand_descriptions=ligand_description_list, 
                               receptor_radius=confidence_args.receptor_radius,
                               cache_path=args.cache_path,
                               remove_hs=confidence_args.remove_hs, max_lig_size=None, c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
                               matching=False, keep_original=False,
                               popsize=confidence_args.matching_popsize,
                               maxiter=confidence_args.matching_maxiter,
                               all_atoms=confidence_args.all_atoms,
                               atom_radius=confidence_args.atom_radius,
                               atom_max_neighbors=confidence_args.atom_max_neighbors,
                               esm_embeddings_path= args.esm_embeddings_path, require_ligand=True,
                               num_workers=args.num_workers)
else:
    confidence_test_dataset = None

t_to_sigma = partial(t_to_sigma_compl, args=score_model_args, linear_tr_schedule=args.init_pocket_center)
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
for idx, orig_complex_graph in tqdm(enumerate(test_loader), desc="Generating Docking Conformation", total=len(test_dataset)): # batch size fixed is 1, because of the randomize_position function
    # if not orig_complex_graph.success[0]:
    #     skipped += 1
    #     print(f"HAPPENING | The test dataset did not contain {test_dataset.complex_names[idx]} for {test_dataset.ligand_descriptions[idx]} and {test_dataset.protein_files[idx]}. We are skipping this complex.")
    #     continue
    try:
        if confidence_test_dataset is not None and args.no_chi_angle:
            confidence_complex_graph = confidence_test_dataset[idx]
            # if not confidence_complex_graph.success:
            #     skipped += 1
            #     print(f"HAPPENING | The confidence dataset did not contain {orig_complex_graph.name}. We are skipping this complex.")
            #     continue
            confidence_data_list = [copy.deepcopy(confidence_complex_graph) for _ in range(N)]
        else:
            confidence_data_list = None
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
        randomize_position(data_list, score_model_args.no_torsion, False, args.no_chi_noise, args.init_pocket_center,
                            score_model_args.tr_sigma_max, score_model_args.atom_radius, score_model_args.atom_max_neighbors)
        lig = orig_complex_graph.mol[0] # just use meta information not the coordinates
        if not args.no_chi_angle and not args.apo_structure:
            true_pockect = orig_complex_graph['sidechain']
            # restore the original pocket center
            true_pockect.node_position = true_pockect.node_position + orig_complex_graph.original_center
        # initialize visualisation
        # pdb = None
        if args.save_visualisation:
            visualization_list = []
            for graph in data_list:
                pdb = PDBFile(lig)
                pdb.add(lig, 0, 0)
                pdb.add((orig_complex_graph['ligand'].pos + orig_complex_graph.original_center).detach().cpu(), 1, 0)
                pdb.add((graph['ligand'].pos + graph.original_center).detach().cpu(), part=1, order=1)
                visualization_list.append(pdb)
        else:
            visualization_list = None

        # run reverse diffusion
        data_list, confidence = sampling(data_list=data_list, model=model,
                                            inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                                            tr_schedule=tr_schedule, rot_schedule=tr_schedule, tor_schedule=tr_schedule, chi_schedule=tr_schedule,
                                            device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
                                            visualization_list=visualization_list, confidence_model=confidence_model,
                                            confidence_data_list=confidence_data_list, confidence_model_args=confidence_args,
                                            batch_size=args.batch_size, no_final_step_noise=args.no_final_step_noise, no_chi_angle=args.no_chi_angle)
        ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for complex_graph in data_list])
        if not args.no_chi_angle:
            protein_atom_pos = np.asarray([complex_graph['atom'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for complex_graph in data_list])
        # reorder predictions based on confidence output
        if confidence is not None and isinstance(confidence_args.rmsd_classification_cutoff, list):
            confidence = confidence[:, 0]
        if confidence is not None:
            confidence = confidence.cpu().numpy()
            re_order = np.argsort(confidence)[::-1]
            confidence = confidence[re_order]
            ligand_pos = ligand_pos[re_order]
            if not args.no_chi_angle:
                protein_atom_pos = protein_atom_pos[re_order]

        # save predictions
        protein_path = orig_complex_graph["name"][0].split('____')[0]
        ligand_description = orig_complex_graph["name"][0].split('____')[-1]
        complex_name = os.path.split(os.path.dirname(protein_path))[1]
        # pdb.set_trace()
        write_dir = f'{args.out_dir}/{complex_name}'
        for rank, pos in enumerate(ligand_pos):
            mol_pred = copy.deepcopy(lig)
            if score_model_args.remove_hs: mol_pred = RemoveHs(mol_pred)
            if rank == 0: write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}.sdf'))
            write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}_confidence{confidence[rank]:.2f}.sdf'))

        if not args.no_chi_angle:
            if not args.apo_structure:
                pickle.dump(true_pockect, open(os.path.join(write_dir, f'true_pocket.pkl'), 'wb')) # save the true pocket object
            for rank, pos in enumerate(protein_atom_pos):
                # read protein_path and ligand_description from the complex_name, to prevent the error of index shift when preprocessing in PDBBind class failded in some cases
                mod_prot = ModifiedPDB(pdb_path=protein_path, ligand_description=ligand_description, pocket_pos=pos)
                if rank == 0: 
                    pickle.dump(pos, open(os.path.join(write_dir, f'rank{rank+1}_pocket_coords.pkl'), 'wb')) # save the predicted pocket comformation
                    mod_prot.to_pdb(os.path.join(write_dir, f'rank{rank+1}_protein.pdb')) # save the predicted protein comformation(with pocket sidechain modification)
                pickle.dump(pos, open(os.path.join(write_dir, f'rank{rank+1}_pocket_coords_confidence{confidence[rank]:.2f}.pkl'), 'wb')) # save the predicted pocket comformation
                mod_prot.to_pdb(os.path.join(write_dir, f'rank{rank+1}_protein_confidence{confidence[rank]:.2f}.pdb')) # save the predicted protein comformation(with pocket sidechain modification)

        # save visualisation frames
        if args.save_visualisation:
            if confidence is not None:
                for rank, batch_idx in enumerate(re_order):
                    visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))
            else:
                for rank, batch_idx in enumerate(ligand_pos):
                    visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))

    except Exception as e:
        print("Failed on", orig_complex_graph["name"], e)
        failures += 1

print(f'Failed for {failures} complexes')
print(f'Skipped {skipped} complexes')
print(f'Results are in {args.out_dir}')