# small script to extract the ligand and save it in a separate file because GNINA will use the ligand position as initial pose
import os
from argparse import FileType, ArgumentParser
import torch
import numpy as np
from biopandas.pdb import PandasPdb
from rdkit import Chem

from tqdm import tqdm
import re
from datasets.pdbbind import read_mol
from datasets.process_mols import read_molecule
from utils.utils import read_strings_from_txt, get_symmetry_rmsd, get_pocket_rmsd
import pickle
import pdb
import wandb

parser = ArgumentParser()
parser.add_argument('--config', type=FileType(mode='r'), default=None)
parser.add_argument('--data_dir', type=str, default='data/PDBBind_processed', help='')
parser.add_argument('--results_path', type=str, default='results/user_predictions_testset', help='Path to folder with trained model and hyperparameters')
parser.add_argument('--file_suffix', type=str, default='_baseline_ligand.pdb', help='Path to folder with trained model and hyperparameters')
parser.add_argument('--file_to_exclude', type=str, default=None, help='')
parser.add_argument('--all_dirs_in_results', action='store_true', default=True, help='Evaluate all directories in the results path instead of using directly looking for the names')
parser.add_argument('--num_predictions', type=int, default=10, help='')
parser.add_argument('--no_id_in_filename', action='store_true', default=False, help='')
parser.add_argument('--test_names_path', type=str, default='data/splits/timesplit_test', help='Path to text file with the folder names in the test set')
parser.add_argument('--no_overlap_names_path', type=str, default='data/splits/timesplit_test_no_rec_overlap', help='Path text file with the folder names in the test set that have no receptor overlap with the train set')
parser.add_argument('--log_dir', type=str, default='workdir/wandb', help='dir for wandb logs')
parser.add_argument('--run_name', type=str, default='ReDockv1_Holo_noise', help='experiment name')
parser.add_argument('--group', type=str, default='Holo_noise', help='experiment group')
parser.add_argument('--project', type=str, default='ReDock', help='')
parser.add_argument('--wandb', action='store_true', default=False, help='Use wandb to log results')
args = parser.parse_args()

print('Reading paths and names.')
names = read_strings_from_txt(args.test_names_path)
names_no_rec_overlap = read_strings_from_txt(args.no_overlap_names_path)
results_path_containments = os.listdir(args.results_path)

all_times = []
successful_names_list = []
rmsds_list = []
sidechain_rmsds_list = []
residue_rmsds_list = []
centroid_distances_list = []
min_cross_distances_list = []
min_self_distances_list = []
without_rec_overlap_list = []

if args.wandb:
    wandb.init(
        entity='kirito_asuna',
        dir=args.log_dir,
        resume='allow',
        project=args.project,
        name=args.run_name,
        id=args.run_name,
        group=args.group,
        config=args
    )

for i, name in enumerate(tqdm(names)):
    actual_num_predictions = args.num_predictions
    mol = read_mol(args.data_dir, name, remove_hs=True)
    mol = Chem.RemoveAllHs(mol)
    orig_ligand_pos = np.array(mol.GetConformer().GetPositions())

    # Find the directory with the name of the complex and read the ligand and pocket conformations
    if args.all_dirs_in_results:
        directory_with_name_list = [directory for directory in results_path_containments if name in directory]
        if directory_with_name_list == []:
            print('Did not find a directory for ', name, '. We are skipping that complex')
            continue
        # if directory is empty, we skip it
        elif len(os.listdir(os.path.join(args.results_path, directory_with_name_list[0]))) == 0:
            print('Directory ', directory_with_name_list[0], ' is empty. We are skipping that complex')
            continue
        else:
            directory_with_name = directory_with_name_list[0]
        ligand_pos = []
        pocket_pos = []
        debug_paths = []
        result_path = os.path.join(args.results_path, directory_with_name)
        file_paths = sorted(os.listdir(result_path))
        if args.file_to_exclude is not None:
            file_paths = [path for path in file_paths if not args.file_to_exclude in path]
        true_pocket = pickle.load(open(os.path.join(result_path, 'true_pockect.pkl'), 'rb'))
        for i in range(args.num_predictions):
            ligand_pttn = re.compile(rf'rank{i+1}_.*\.sdf$')
            pocket_pttn = re.compile(rf'rank{i+1}_.*\.pkl$')
            file_path = [path for path in file_paths if ligand_pttn.match(path)][0]
            pckt_path = [path for path in file_paths if pocket_pttn.match(path)][0]
            try:
                mol_pred = read_molecule(os.path.join(args.results_path, directory_with_name, file_path),remove_hs=True, sanitize=True)
                mol_pred = Chem.RemoveAllHs(mol_pred)
                pckt_pred_coords = pickle.load(open(os.path.join(result_path, pckt_path), 'rb'))
            except:
                error_file = os.path.join(args.results_path, directory_with_name, file_path)
                print('Could not read ', error_file, '. We are skipping that prediction')
                actual_num_predictions = actual_num_predictions - 1
                print('actual_num_predictions: ', actual_num_predictions)
                if len(ligand_pos) == 0:
                    ligand_pos.append(orig_ligand_pos)
                    pocket_pos.append(true_pocket.node_positions)
                else:
                    ligand_pos.append(ligand_pos[-1])
                    pocket_pos.append(pocket_pos[-1])
                continue
            ligand_pos.append(mol_pred.GetConformer().GetPositions())
            pocket_pos.append(torch.from_numpy(pckt_pred_coords))
            debug_paths.append(file_path)
    
        ligand_pos = np.asarray(ligand_pos)
    else: #TODO: add support for single files with pocket evaluation
        if not os.path.exists(os.path.join(args.results_path, name, f'{"" if args.no_id_in_filename else name}{args.file_suffix}')): raise Exception('path did not exists:', os.path.join(args.results_path, name, f'{"" if args.no_id_in_filename else name}{args.file_suffix}'))
        mol_pred = read_molecule(os.path.join(args.results_path, name, f'{"" if args.no_id_in_filename else name}{args.file_suffix}'), remove_hs=True, sanitize=True)
        if mol_pred == None:
            print("Skipping ", name, ' because RDKIT could not read it.')
            continue
        mol_pred = Chem.RemoveAllHs(mol_pred)
        ligand_pos = np.asarray([np.array(mol_pred.GetConformer(i).GetPositions()) for i in range(args.num_predictions)])
    
    # Calculate Metrics For one ligand and pocket complex with num_predictions predictions
    sc_rmsd, residue_rmsd = get_pocket_rmsd(true_pocket, pocket_pos) # a list of rmsd like the ligand rmsd
    try:
        rmsd = get_symmetry_rmsd(mol, orig_ligand_pos, [l for l in ligand_pos], mol_pred)
    except Exception as e:
        print("Using non corrected RMSD because of the error:", e)
        rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))

    # Stat 1 - RMSD
    rmsds_list.append(rmsd) # rmsd is [num_predictions], rmsds_list is [num_complexes, num_predictions] 
    sidechain_rmsds_list.append(sc_rmsd)
    residue_rmsds_list.append(residue_rmsd)
    # Stat 2- Liagnd Center Distance
    centroid_distances_list.append(np.linalg.norm(ligand_pos.mean(axis=1) - orig_ligand_pos[None,:].mean(axis=1), axis=1))
    # Stat 3 - Min Cross Distance and sekf distance
    rec_path = os.path.join(args.data_dir, name, f'{name}_protein_processed.pdb')
    if not os.path.exists(rec_path):
        rec_path = os.path.join(args.data_dir, name,f'{name}_protein_obabel_reduce.pdb')
    rec = PandasPdb().read_pdb(rec_path)
    rec_df = rec.df['ATOM']
    receptor_pos = rec_df[['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)
    receptor_pos = np.tile(receptor_pos, (args.num_predictions, 1, 1))

    cross_distances = np.linalg.norm(receptor_pos[:, :, None, :] - ligand_pos[:, None, :, :], axis=-1)
    self_distances = np.linalg.norm(ligand_pos[:, :, None, :] - ligand_pos[:, None, :, :], axis=-1)
    self_distances =  np.where(np.eye(self_distances.shape[2]), np.inf, self_distances)
    min_cross_distances_list.append(np.min(cross_distances, axis=(1,2)))
    min_self_distances_list.append(np.min(self_distances, axis=(1, 2)))
    successful_names_list.append(name)
    without_rec_overlap_list.append(1 if name in names_no_rec_overlap else 0)

# Calculate performance metrics
top1_performance_metrics = {}
top5_performance_metrics = {}
top10_performance_metrics = {}
report_metrics = {}
logs = {}
for overlap in ['', 'no_overlap_']:
    if 'no_overlap_' == overlap:
        without_rec_overlap = np.array(without_rec_overlap_list, dtype=bool)
        rmsds = np.array(rmsds_list)[without_rec_overlap] # shape [num_complexes, num_predictions]
        centroid_distances = np.array(centroid_distances_list)[without_rec_overlap]
        min_cross_distances = np.array(min_cross_distances_list)[without_rec_overlap]
        min_self_distances = np.array(min_self_distances_list)[without_rec_overlap]
        successful_names = np.array(successful_names_list)[without_rec_overlap]
        sc_rmsds = np.array(sidechain_rmsds_list)[without_rec_overlap]
        residue_rmsds = np.array(residue_rmsds_list)[without_rec_overlap]
    else:
        rmsds = np.array(rmsds_list) # shape [num_complexes, num_predictions]
        centroid_distances = np.array(centroid_distances_list)
        min_cross_distances = np.array(min_cross_distances_list)
        min_self_distances = np.array(min_self_distances_list)
        successful_names = np.array(successful_names_list)
        sc_rmsds = np.array(sidechain_rmsds_list)
        residue_rmsds = np.array(residue_rmsds_list)
# Save results for visualization and further analysis
    np.save(os.path.join(args.results_path, f'{overlap}names.npy'), successful_names)
    np.save(os.path.join(args.results_path, f'{overlap}rmsds.npy'), rmsds)
    np.save(os.path.join(args.results_path, f'{overlap}sc_rmsds.npy'), sc_rmsds)
    np.save(os.path.join(args.results_path, f'{overlap}residue_rmsds.npy'), residue_rmsds)
    np.save(os.path.join(args.results_path, f'{overlap}centroid_distances.npy'), centroid_distances)
    np.save(os.path.join(args.results_path, f'{overlap}min_cross_distances.npy'), np.array(min_cross_distances))
    np.save(os.path.join(args.results_path, f'{overlap}min_self_distances.npy'), np.array(min_self_distances))

# Top 1 rmsd, sc_rmsd, residue_rmsd and centroid distance
    top1_performance_metrics.update({
        f'{overlap}steric_clash_fraction': (100 * (min_cross_distances < 0.4).sum() / len(min_cross_distances) / args.num_predictions).__round__(2),
        f'{overlap}self_intersect_fraction': (100 * (min_self_distances < 0.4).sum() / len(min_self_distances) / args.num_predictions).__round__(2),

        f'{overlap}mean_rmsd': rmsds[:,0].mean(),
        f'{overlap}rmsds_percentile_25': np.percentile(rmsds[:,0], 25).round(2),
        f'{overlap}rmsds_percentile_75': np.percentile(rmsds[:,0], 75).round(2),

        f'{overlap}mean_sc_rmsd': sc_rmsds[:,0].mean(),
        f'{overlap}sc_rmsds_percentile_25': np.percentile(sc_rmsds[:,0], 25).round(2),
        f'{overlap}sc_rmsds_percentile_75': np.percentile(sc_rmsds[:,0], 75).round(2),

        f'{overlap}mean_residue_rmsd': residue_rmsds[:,0].mean(),
        f'{overlap}residue_rmsds_below_2': (100 * (residue_rmsds[:,0] < 2).sum() / len(residue_rmsds[:,0])),
        f'{overlap}residue_rmsds_below_5': (100 * (residue_rmsds[:,0] < 5).sum() / len(residue_rmsds[:,0])),
        f'{overlap}residue_rmsds_percentile_25': np.percentile(residue_rmsds[:,0], 25).round(2),
        f'{overlap}residue_rmsds_percentile_50': np.percentile(residue_rmsds[:,0], 50).round(2), # median
        f'{overlap}residue_rmsds_percentile_75': np.percentile(residue_rmsds[:,0], 75).round(2),

        f'{overlap}mean_centroid': centroid_distances[:,0].mean().__round__(2),
        f'{overlap}centroid_below_2': (100 * (centroid_distances[:,0] < 2).sum() / len(centroid_distances[:,0])).__round__(2),
        f'{overlap}centroid_below_5': (100 * (centroid_distances[:,0] < 5).sum() / len(centroid_distances[:,0])).__round__(2),
        f'{overlap}centroid_percentile_25': np.percentile(centroid_distances[:,0], 25).round(2),
        f'{overlap}centroid_percentile_50': np.percentile(centroid_distances[:,0], 50).round(2),
        f'{overlap}centroid_percentile_75': np.percentile(centroid_distances[:,0], 75).round(2),
    })
    if overlap == '':
        report_metrics.update({
            f'{overlap}rmsds_below_2': (100 * (rmsds[:,0] < 2).sum() / len(rmsds[:,0])),
            f'{overlap}rmsds_below_5': (100 * (rmsds[:,0] < 5).sum() / len(rmsds[:,0])),
            f'{overlap}rmsds_percentile_50': np.percentile(rmsds[:,0], 50).round(2), # median
            f'{overlap}sc_rmsds_below_1': (100 * (sc_rmsds[:,0] < 1).sum() / len(sc_rmsds[:,0])),
            f'{overlap}sc_rmsds_below_2': (100 * (sc_rmsds[:,0] < 2).sum() / len(sc_rmsds[:,0])),
            f'{overlap}sc_rmsds_percentile_50': np.percentile(sc_rmsds[:,0], 50).round(2), # median
        })  
# Best of Top 5 rmsd, sc_rmsd, residue_rmsd and centroid distance
    top5_rmsds = np.min(rmsds[:, :5], axis=1)
    top5_sc_rmsds = np.min(sc_rmsds[:, :5], axis=1)
    top5_residue_rmsds = np.min(residue_rmsds[:, :5], axis=1)
    top5_centroid_distances = centroid_distances[np.arange(rmsds.shape[0])[:,None],np.argsort(rmsds[:, :5], axis=1)][:,0]
    top5_min_cross_distances = min_cross_distances[np.arange(rmsds.shape[0])[:,None],np.argsort(rmsds[:, :5], axis=1)][:,0]
    top5_min_self_distances = min_self_distances[np.arange(rmsds.shape[0])[:,None],np.argsort(rmsds[:, :5], axis=1)][:,0]
    
    top5_performance_metrics.update({
        f'{overlap}top5_steric_clash_fraction': (100 * (top5_min_cross_distances < 0.4).sum() / len(top5_min_cross_distances)).__round__(2),
        f'{overlap}top5_self_intersect_fraction': (100 * (top5_min_self_distances < 0.4).sum() / len(top5_min_self_distances)).__round__(2),

        f'{overlap}top5_mean_rmsd': top5_rmsds[:,0].mean(),
        f'{overlap}top5_rmsds_percentile_25': np.percentile(top5_rmsds, 25).round(2),
        f'{overlap}top5_rmsds_percentile_75': np.percentile(top5_rmsds, 75).round(2),

        f'{overlap}top5_mean_sc_rmsd': top5_sc_rmsds[:,0].mean(),
        f'{overlap}top5_sc_rmsds_percentile_25': np.percentile(top5_sc_rmsds, 25).round(2),
        f'{overlap}top5_sc_rmsds_percentile_75': np.percentile(top5_sc_rmsds, 75).round(2),

        f'{overlap}top5_mean_residue_rmsd': top5_residue_rmsds[:,0].mean(),
        f'{overlap}top5_residue_rmsds_below_2': (100 * (top5_residue_rmsds < 2).sum() / len(top5_residue_rmsds)).__round__(2),
        f'{overlap}top5_residue_rmsds_below_5': (100 * (top5_residue_rmsds < 5).sum() / len(top5_residue_rmsds)).__round__(2),
        f'{overlap}top5_residue_rmsds_percentile_25': np.percentile(top5_residue_rmsds, 25).round(2),
        f'{overlap}top5_residue_rmsds_percentile_50': np.percentile(top5_residue_rmsds, 50).round(2),
        f'{overlap}top5_residue_rmsds_percentile_75': np.percentile(top5_residue_rmsds, 75).round(2),

        f'{overlap}top5_mean_centroid': top5_centroid_distances[:,0].mean().__round__(2), # TODO: check if this is correct
        f'{overlap}top5_centroid_below_2': (100 * (top5_centroid_distances < 2).sum() / len(top5_centroid_distances)).__round__(2),
        f'{overlap}top5_centroid_below_5': (100 * (top5_centroid_distances < 5).sum() / len(top5_centroid_distances)).__round__(2),
        f'{overlap}top5_centroid_percentile_25': np.percentile(top5_centroid_distances, 25).round(2),
        f'{overlap}top5_centroid_percentile_50': np.percentile(top5_centroid_distances, 50).round(2),
        f'{overlap}top5_centroid_percentile_75': np.percentile(top5_centroid_distances, 75).round(2),
    })

    if overlap == '':
        report_metrics.update({
            f'{overlap}top5_rmsds_below_2': (100 * (top5_rmsds < 2).sum() / len(top5_rmsds)),
            f'{overlap}top5_rmsds_below_5': (100 * (top5_rmsds < 5).sum() / len(top5_rmsds)),
            f'{overlap}top5_rmsds_percentile_50': np.percentile(top5_rmsds, 50).round(2), # median
            f'{overlap}top5_sc_rmsds_below_1': (100 * (top5_sc_rmsds < 1).sum() / len(top5_sc_rmsds)),
            f'{overlap}top5_sc_rmsds_below_2': (100 * (top5_sc_rmsds < 2).sum() / len(top5_sc_rmsds)),
            f'{overlap}top5_sc_rmsds_percentile_50': np.percentile(top5_sc_rmsds, 50).round(2), # median
        })

# Best of Top 10
    top10_rmsds = np.min(rmsds[:, :10], axis=1)
    top10_sc_rmsds = np.min(sc_rmsds[:, :10], axis=1)
    top10_residue_rmsds = np.min(residue_rmsds[:, :10], axis=1)
    top10_centroid_distances = centroid_distances[np.arange(rmsds.shape[0])[:,None],np.argsort(rmsds[:, :10], axis=1)][:,0]
    top10_min_cross_distances = min_cross_distances[np.arange(rmsds.shape[0])[:,None],np.argsort(rmsds[:, :10], axis=1)][:,0]
    top10_min_self_distances = min_self_distances[np.arange(rmsds.shape[0])[:,None],np.argsort(rmsds[:, :10], axis=1)][:,0]
    top10_performance_metrics.update({
        f'{overlap}top10_self_intersect_fraction': (100 * (top10_min_self_distances < 0.4).sum() / len(top10_min_self_distances)).__round__(2),
        f'{overlap}top10_steric_clash_fraction': ( 100 * (top10_min_cross_distances < 0.4).sum() / len(top10_min_cross_distances)).__round__(2),

        f'{overlap}top10_mean_rmsd': top10_rmsds[:,0].mean(),
        f'{overlap}top10_rmsds_below_2': (100 * (top10_rmsds < 2).sum() / len(top10_rmsds)).__round__(2),
        f'{overlap}top10_rmsds_below_5': (100 * (top10_rmsds < 5).sum() / len(top10_rmsds)).__round__(2),
        f'{overlap}top10_rmsds_percentile_25': np.percentile(top10_rmsds, 25).round(2),
        f'{overlap}top10_rmsds_percentile_50': np.percentile(top10_rmsds, 50).round(2),
        f'{overlap}top10_rmsds_percentile_75': np.percentile(top10_rmsds, 75).round(2),
        
        f'{overlap}top10_mean_sc_rmsd': top10_sc_rmsds[:,0].mean(),
        f'{overlap}top10_sc_rmsds_below_1': (100 * (top10_sc_rmsds < 1).sum() / len(top10_sc_rmsds)).__round__(2),
        f'{overlap}top10_sc_rmsds_below_2': (100 * (top10_sc_rmsds < 2).sum() / len(top10_sc_rmsds)).__round__(2),
        f'{overlap}top10_sc_rmsds_percentile_25': np.percentile(top10_sc_rmsds, 25).round(2),
        f'{overlap}top10_sc_rmsds_percentile_50': np.percentile(top10_sc_rmsds, 50).round(2),
        f'{overlap}top10_sc_rmsds_percentile_75': np.percentile(top10_sc_rmsds, 75).round(2),

        f'{overlap}top10_mean_residue_rmsd': top10_residue_rmsds[:,0].mean(),
        f'{overlap}top10_residue_rmsds_below_2': (100 * (top10_residue_rmsds < 2).sum() / len(top10_residue_rmsds)).__round__(2),
        f'{overlap}top10_residue_rmsds_below_5': (100 * (top10_residue_rmsds < 5).sum() / len(top10_residue_rmsds)).__round__(2),
        f'{overlap}top10_residue_rmsds_percentile_25': np.percentile(top10_residue_rmsds, 25).round(2),
        f'{overlap}top10_residue_rmsds_percentile_50': np.percentile(top10_residue_rmsds, 50).round(2),
        f'{overlap}top10_residue_rmsds_percentile_75': np.percentile(top10_residue_rmsds, 75).round(2),

        f'{overlap}top10_mean_centroid': top10_centroid_distances[:,0].mean().__round__(2), # TODO: check if this is correct
        f'{overlap}top10_centroid_below_2': (100 * (top10_centroid_distances < 2).sum() / len(top10_centroid_distances)).__round__(2),
        f'{overlap}top10_centroid_below_5': (100 * (top10_centroid_distances < 5).sum() / len(top10_centroid_distances)).__round__(2),
        f'{overlap}top10_centroid_percentile_25': np.percentile(top10_centroid_distances, 25).round(2),
        f'{overlap}top10_centroid_percentile_50': np.percentile(top10_centroid_distances, 50).round(2),
        f'{overlap}top10_centroid_percentile_75': np.percentile(top10_centroid_distances, 75).round(2),
    })
for k in top1_performance_metrics:
    print(k, top1_performance_metrics[k])
    logs['top1_metric/' + k] = top1_performance_metrics[k]
print("-----------------------------------------------------------------------------")
for k in top5_performance_metrics:
    print(k, top5_performance_metrics[k])
    logs['top5_metric/' + k] = top5_performance_metrics[k]
print("-----------------------------------------------------------------------------")
for k in top10_performance_metrics:
    # print(k, top10_performance_metrics[k])
    logs['top10_metric/' + k] = top10_performance_metrics[k]

for k in report_metrics:
    logs['report_metrics/' + k] = report_metrics[k]

if args.wandb:
    wandb.log(logs)
    

