import numpy as np
import torch
from torch_geometric.loader import DataLoader

from utils.diffusion_utils import modify_conformer, set_time
from utils.torsion import modify_conformer_torsion_angles
from scipy.spatial.transform import Rotation as R
from utils.rotamer import rotate_side_chain
from torch_cluster import radius_graph
from utils import rotamer
import copy
import pdb
NUM_CHI_ANGLES = 4

#NOTE: Has been modified to handle batch in ReZero_Docking.py
def randomize_position(data_list, no_torsion, no_random, no_sidechain, tr_sigma_max, 
                       atom_radius, atom_max_neighbors=None):
    # in place modification of the list
    if not no_torsion:
        # randomize torsion angles
        for complex_graph in data_list:
            torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=complex_graph['ligand'].edge_mask.sum())
            if isinstance(complex_graph['ligand'].mask_rotate, list):
                mask_rotate = complex_graph['ligand'].mask_rotate[0]
            else:
                mask_rotate = complex_graph['ligand'].mask_rotate
            complex_graph['ligand'].pos = \
                modify_conformer_torsion_angles(complex_graph['ligand'].pos,
                                                complex_graph['ligand','lig_bond','ligand'].edge_index.T[
                                                complex_graph['ligand'].edge_mask],
                                                mask_rotate, torsion_updates)

    for complex_graph in data_list:
        # randomize position
        molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        random_rotation = torch.from_numpy(R.random().as_matrix()).float()
        complex_graph['ligand'].pos = (complex_graph['ligand'].pos - molecule_center) @ random_rotation.T #NOTE: the ligand center now is (0,0,0) the same with pocket Ca center (see pdbbind.py line 480)
        # base_rmsd = np.sqrt(np.sum((complex_graph['ligand'].pos.cpu().numpy() - orig_complex_graph['ligand'].pos.numpy()) ** 2, axis=1).mean())
        # tr_sigma_max = 5 # restrict the initial position of ligand in the pocket
        # pocket_center = torch.mean(complex_graph['atom'].pos, dim=0, keepdim=True) # this caculation is meaningless as the pocket all atom center is also near zero.
        # complex_graph['ligand'].pos += pocket_center
        # pdb.set_trace()
        # tr_sigma_max = 1 # change to 1 for initializing ligand near pocket center and diversify the initial position
        #NOTE: Can't be two small, the actual ligand center may not be near the pocket center
        if not no_random:  # note for now the torsion angles are still randomised
            tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))
            complex_graph['ligand'].pos += tr_update
        if not no_sidechain:
            protein = complex_graph['sidechain']
            # pdb.set_trace()
            torsion_updates = torch.rand((protein.num_residue, 4), device=protein.node_position.device) * 2 * np.pi
            rotate_side_chain(protein, torsion_updates)
            # Modify the complex graph according to protein dict
            complex_graph['atom'].pos = protein.node_position
            atom_coords = protein.node_position
            atoms_edge_index = radius_graph(atom_coords, atom_radius,
                                           max_num_neighbors=atom_max_neighbors if atom_max_neighbors else 1000)
            complex_graph['atom', 'atom_contact', 'atom'].edge_index = atoms_edge_index
            

def sampling(data_list, model, inference_steps, tr_schedule, rot_schedule, tor_schedule, chi_schedule, device, t_to_sigma, model_args,
             no_random=False, ode=False, no_chi_angle=False, visualization_list=None, confidence_model=None, confidence_data_list=None,
             confidence_model_args=None, batch_size=32, no_final_step_noise=False):
    N = len(data_list)

    for t_idx in range(inference_steps):
        t_tr, t_rot, t_tor, t_chi = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx], chi_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
        dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]
        dt_chi = chi_schedule[t_idx] - chi_schedule[t_idx + 1] if t_idx < inference_steps - 1 else chi_schedule[t_idx]

        loader = DataLoader(data_list, batch_size=batch_size)
        new_data_list = []
        # batch_add = False
        # pdb.set_trace()
        for complex_graph_batch in loader:
            # pdb.set_trace()
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)

            if model_args.all_atoms: # handle batch when including sidechain graph
                batch_index = complex_graph_batch['atom'].batch # [num_atoms]
                protein = complex_graph_batch['sidechain']
                # calculate the offset of each protein
                offset = 0
                # check if protein.atom2residue is already batched
                # num_residue1 = protein.num_residue[0]
                if b != 1: # only handle mini-batch
                    end_last = protein.atom2residue[batch_index == 0][-1]
                    begin_first = protein.atom2residue[batch_index == 1][0]
                    if end_last + 1 != begin_first:
                        for i in range(b):
                            protein.atom2residue[batch_index == i] += offset
                            offset += protein.num_residue[i]
            # pdb.set_trace()
            # causal sidechain mask
            # protein.num_nodes is automatically calculated by pytorch geometric, don't need sum(protein.num_residue)
            # protein.num_residue = sum(protein.num_residue) # handle batch, but there is only one sample and this operation will result in problem with splitting batch
            # protein.num_nodes = protein.num_residue # fix the bug with batching num_nodes property
            tr_sigma, rot_sigma, tor_sigma, chi_sigma = t_to_sigma(t_tr, t_rot, t_tor, t_chi)
            set_time(complex_graph_batch, t_tr, t_rot, t_tor, t_chi, b, model_args.all_atoms, device)
            # pdb.set_trace()
            with torch.no_grad():
                if not no_chi_angle:
                # tr_score, rot_score, tor_score, chi_score = model(complex_graph_batch)
                    for chi_id in range(NUM_CHI_ANGLES):
                        # pdb.set_trace()
                        chis = rotamer.get_chis(protein, protein.node_position) # all chi angles including currently unchanged angles
                        # predict score for each chi angle, tr, rot and tor score will also be predicted autoregressively
                        #NOTE: in My implementation remove_by_chi will modify the complex_graph_batch directly for avoiding deep copy in training
                        chi_protein_bacth = copy.deepcopy(complex_graph_batch)
                        chi_protein_batch = rotamer.remove_by_chi(chi_protein_bacth, chi_id)
                        chi_protein_batch.chi_id = chi_id
                        tr_score, rot_score, tor_score, chi_score = model.predict(chi_protein_batch, sampling=True)
                        # chi_score = chi_pred[0]
                        # step backward for chi angle and predict next chi. It could be noisy but I have no choise
                        chis = model.so2_periodic[0].step(chis, chi_score, chi_sigma, dt_chi, chi_protein_bacth['sidechain'].chi_1pi_periodic_mask)
                        chis = model.so2_periodic[1].step(chis, chi_score, chi_sigma, dt_chi, chi_protein_bacth['sidechain'].chi_2pi_periodic_mask)
                        # pdb.set_trace()
                        protein = rotamer.set_chis(protein, chis)
                        # Modify the complex graph according to protein dict
                        complex_graph_batch['atom'].pos = protein.node_position
                        atom_coords = protein.node_position
                        atoms_edge_index = radius_graph(atom_coords, model_args.atom_radius, complex_graph_batch['atom'].batch,
                                                    max_num_neighbors=model_args.atom_max_neighbors if model_args.atom_max_neighbors else 1000)
                        complex_graph_batch['atom', 'atom_contact', 'atom'].edge_index = atoms_edge_index
                elif model_args.all_atoms:
                    tr_score, rot_score, tor_score, _ = model(complex_graph_batch)
                else:
                    tr_score, rot_score, tor_score = model(complex_graph_batch) # For original DiffDock                    

            tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))
            rot_g = 2 * rot_sigma * torch.sqrt(torch.tensor(np.log(model_args.rot_sigma_max / model_args.rot_sigma_min)))
            # pdb.set_trace()
            if ode:
                tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score.cpu()).cpu()
                rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g ** 2).cpu()
            else:
                tr_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()

                rot_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(b, 3))
                rot_perturb = (rot_score.cpu() * dt_rot * rot_g ** 2 + rot_g * np.sqrt(dt_rot) * rot_z).cpu()

            if not model_args.no_torsion:
                tor_g = tor_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tor_sigma_max / model_args.tor_sigma_min)))
                if ode:
                    tor_perturb = (0.5 * tor_g ** 2 * dt_tor * tor_score.cpu()).numpy()
                else:
                    tor_z = torch.zeros(tor_score.shape) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                        else torch.normal(mean=0, std=1, size=tor_score.shape)
                    tor_perturb = (tor_g ** 2 * dt_tor * tor_score.cpu() + tor_g * np.sqrt(dt_tor) * tor_z).numpy()
                # if not complex_graph_batch:
                #     torsions_per_molecule = tor_perturb.shape[0] // b #NOTE: It assumes that the torsion angles are the same for each molecule in the batch
                torsion_index = complex_graph_batch['ligand'].batch[complex_graph_batch['ligand','lig_bond','ligand'].edge_index[0][complex_graph_batch['ligand'].edge_mask]]
                torsion_index = torsion_index.cpu().numpy()
            else:
                tor_perturb = None

            # Apply noise
            new_data_list.extend([modify_conformer(complex_graph, tr_perturb[i:i + 1], rot_perturb[i:i + 1].squeeze(0),
                                          tor_perturb[torsion_index == i] if not model_args.no_torsion else None)
                         for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())])
        data_list = new_data_list

        if visualization_list is not None:
            for idx, visualization in enumerate(visualization_list):
                visualization.add((data_list[idx]['ligand'].pos + data_list[idx].original_center).detach().cpu(),
                                  part=1, order=t_idx + 2)

    with torch.no_grad():
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size)
            confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size))
            confidence = []
            # pdb.set_trace()
            for complex_graph_batch in loader:
                complex_graph_batch = complex_graph_batch.to(device)
                if confidence_data_list is not None:
                    confidence_complex_graph_batch = next(confidence_loader).to(device)
                    confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
                    #TODO: Replace the protein sidechain conformation
                    set_time(confidence_complex_graph_batch, 0, 0, 0, 0, N, confidence_model_args.all_atoms, device)
                    confidence.append(confidence_model(confidence_complex_graph_batch))
                else:
                    set_time(complex_graph_batch, 0, 0, 0, 0, N, confidence_model_args.all_atoms, device)
                    confidence.append(confidence_model(complex_graph_batch))
            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None

    return data_list, confidence
