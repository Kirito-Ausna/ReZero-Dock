import copy

import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from confidence.dataset import ListDataset
from utils import so3, torus
from utils.sampling import randomize_position, sampling
import torch
from utils.diffusion_utils import get_t_schedule
from torch_scatter import scatter_mean
from utils import rotamer
from utils.rotamer import atom_name_vocab, _get_symm_atoms, _rmsd_per_residue

NUM_CHI_ANGLES = 4

def loss_function(tr_pred, rot_pred, tor_pred, chi_pred, data, t_to_sigma, device, tr_weight=1, rot_weight=1,
                  tor_weight=1, chi_weight=1, apply_mean=True, no_torsion=False, no_chi_angle=False):
    tr_sigma, rot_sigma, tor_sigma, chi_sigma = t_to_sigma(
        *[torch.cat([d.complex_t[noise_type] for d in data]) if device.type == 'cuda' else data.complex_t[noise_type]
          for noise_type in ['tr', 'rot', 'tor', 'chi']])
    mean_dims = (0, 1) if apply_mean else 1

    # translation component
    tr_score = torch.cat([d.tr_score for d in data], dim=0) if device.type == 'cuda' else data.tr_score
    tr_sigma = tr_sigma.unsqueeze(-1)
    tr_loss = ((tr_pred.cpu() - tr_score) ** 2 * tr_sigma ** 2)
    # set nan in tr_loss to 0
    tr_loss[torch.isnan(tr_loss)] = torch.zeros(1, dtype=torch.float)
    tr_loss = tr_loss.mean(dim=mean_dims)
    tr_base_loss = (tr_score ** 2 * tr_sigma ** 2).mean(dim=mean_dims).detach()
    # rotation component
    rot_score = torch.cat([d.rot_score for d in data], dim=0) if device.type == 'cuda' else data.rot_score
    rot_score_norm = so3.score_norm(rot_sigma.cpu()).unsqueeze(-1)
    rot_loss = (((rot_pred.cpu() - rot_score) / rot_score_norm) ** 2)
    rot_loss[torch.isnan(rot_loss)] = torch.zeros(1, dtype=torch.float)
    rot_loss = rot_loss.mean(dim=mean_dims)
    rot_base_loss = ((rot_score / rot_score_norm) ** 2).mean(dim=mean_dims).detach()
    # torsion component
    if not no_torsion:
        edge_tor_sigma = torch.from_numpy(
            np.concatenate([d.tor_sigma_edge for d in data] if device.type == 'cuda' else data.tor_sigma_edge))
        tor_score = torch.cat([d.tor_score for d in data], dim=0) if device.type == 'cuda' else data.tor_score
        tor_score_norm2 = torch.tensor(torus.score_norm(edge_tor_sigma.cpu().numpy())).float()
        tor_loss = ((tor_pred.cpu() - tor_score) ** 2 / tor_score_norm2)
        tor_base_loss = ((tor_score ** 2 / tor_score_norm2)).detach()
        if apply_mean:
            tor_loss, tor_base_loss = tor_loss.mean() * torch.ones(1, dtype=torch.float), tor_base_loss.mean() * torch.ones(1, dtype=torch.float)
        else:
            index = torch.cat([torch.ones(d['ligand'].edge_mask.sum()) * i for i, d in
                               enumerate(data)]).long() if device.type == 'cuda' else data['ligand'].batch[
                data['ligand', 'ligand'].edge_index[0][data['ligand'].edge_mask]]
            num_graphs = len(data) if device.type == 'cuda' else data.num_graphs
            t_l, t_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(tor_loss.shape))
            c = c + 0.0001
            t_l.index_add_(0, index, tor_loss)
            t_b_l.index_add_(0, index, tor_base_loss)
            tor_loss, tor_base_loss = t_l / c, t_b_l / c
    else:
        if apply_mean:
            tor_loss, tor_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            tor_loss, tor_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(len(rot_loss), dtype=torch.float)
    # pdb.set_trace()
    if not no_chi_angle:
        # chi_score = torch.cat([d.chi_score for d in data], dim=0) if device.type == 'cuda' else data.chi_score
        chi_pred, chi_norm, chi_score = chi_pred
        # chi_mask = torch.cat([d['sidechain'].chi_mask for d in data], dim=0) if device.type == 'cuda' else data['sidechain'].chi_mask
        # because of rotamer.remove_by_chi, the true chi_mask is in model, not in current data
        chi_score = chi_score.cpu()
        chi_norm = chi_norm.cpu()
        chi_loss = ((chi_pred.cpu() - chi_score) ** 2 / chi_norm)
        chi_base_loss = (chi_score ** 2 / chi_norm).detach()
        if apply_mean: # torch.ones is for changing the type float64 to float32
            chi_loss, chi_base_loss = chi_loss.mean() * torch.ones(1, dtype=torch.float), chi_base_loss.mean() * torch.ones(1, dtype=torch.float)
        else:
            atom2graph = torch.cat([i * torch.ones(d['sidechain'].num_residue) for i, d in enumerate(data)]).long() if device.type == 'cuda' else data['receptor'].batch
            chi_loss = scatter_mean(chi_loss, atom2graph, dim=0).mean(-1).float()# [Num_graph]
            chi_base_loss = scatter_mean(chi_base_loss, atom2graph, dim=0).mean(-1).float()# [Num_graph]
            # chi_loss, chi_base_loss = chi_loss.mean() * torch.ones(len(rot_loss), dtype=torch.float), chi_base_loss.mean() * torch.ones(len(rot_loss), dtype=torch.float)
    else:
        if apply_mean:
            chi_loss, chi_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            chi_loss, chi_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(len(rot_loss), dtype=torch.float)
    loss = tr_loss * tr_weight + rot_loss * rot_weight + tor_loss * tor_weight + chi_loss * chi_weight
    return loss, tr_loss.detach(), rot_loss.detach(), tor_loss.detach(), chi_loss.detach(), tr_base_loss, rot_base_loss, tor_base_loss, chi_base_loss

class AverageMeter():
    def __init__(self, types, unpooled_metrics=False, intervals=1):
        self.types = types
        self.intervals = intervals
        self.count = 0 if intervals == 1 else torch.zeros(len(types), intervals)
        self.acc = {t: torch.zeros(intervals) for t in types}
        self.unpooled_metrics = unpooled_metrics

    def add(self, vals, interval_idx=None):
        # pdb.set_trace()
        if self.intervals == 1:
            self.count += 1 if vals[0].dim() == 0 else len(vals[0])
            # pdb.set_trace()
            for type_idx, v in enumerate(vals):
                self.acc[self.types[type_idx]] += v.sum() if self.unpooled_metrics else v
        else:
            for type_idx, v in enumerate(vals):
                self.count[type_idx].index_add_(0, interval_idx[type_idx], torch.ones(len(v)))
                # pdb.set_trace()
                if not torch.allclose(v, torch.tensor(0.0)):
                    self.acc[self.types[type_idx]].index_add_(0, interval_idx[type_idx], v)

    def summary(self):
        if self.intervals == 1:
            out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    out['int' + str(i) + '_' + k] = (
                            list(self.acc.values())[type_idx][i] / self.count[type_idx][i]).item()
            return out
    
def train_epoch(model, loader, optimizer, device, t_to_sigma, loss_fn, ema_weights):
    model.train()
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'chi_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'chi_base_loss'])
    
    for data in tqdm(loader, total=len(loader)):
        if device.type == 'cuda' and len(data) == 1 or device.type == 'cpu' and data.num_graphs == 1:
            print("Skipping batch of size 1 since otherwise batchnorm would not work.")
        optimizer.zero_grad()
        try:
            tr_pred, rot_pred, tor_pred, chi_pred = model(data)
            loss, tr_loss, rot_loss, tor_loss, chi_loss, tr_base_loss, rot_base_loss, tor_base_loss, chi_base_loss = \
                loss_fn(tr_pred, rot_pred, tor_pred, chi_pred, data=data, t_to_sigma=t_to_sigma, device=device)
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema_weights.update(model.parameters())
            meter.add([loss.cpu().detach(), tr_loss, rot_loss, tor_loss, chi_loss, tr_base_loss, rot_base_loss, tor_base_loss, chi_base_loss])
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    return meter.summary()


def test_epoch(model, loader, device, t_to_sigma, loss_fn, test_sigma_intervals=False):
    model.eval()
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'chi_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'chi_base_loss'],
                         unpooled_metrics=True)

    if test_sigma_intervals:
        meter_all = AverageMeter(
            ['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'chi_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'chi_base_loss'],
            unpooled_metrics=True, intervals=10)

    for data in tqdm(loader, total=len(loader)):
        try:
            with torch.no_grad():
                # pdb.set_trace()
                tr_pred, rot_pred, tor_pred, chi_pred = model(data)
                
            loss, tr_loss, rot_loss, tor_loss, chi_loss, tr_base_loss, rot_base_loss, tor_base_loss, chi_base_loss = \
                loss_fn(tr_pred, rot_pred, tor_pred, chi_pred, data=data, t_to_sigma=t_to_sigma, apply_mean=False, device=device)
            meter.add([loss.cpu().detach(), tr_loss, rot_loss, tor_loss, chi_loss, tr_base_loss, rot_base_loss, tor_base_loss, chi_base_loss])

            if test_sigma_intervals > 0:
                complex_t_tr, complex_t_rot, complex_t_tor, complex_t_chi = [torch.cat([d.complex_t[noise_type] for d in data]) for
                                                                noise_type in ['tr', 'rot', 'tor', 'chi']]
                sigma_index_tr = torch.round(complex_t_tr.cpu() * (10 - 1)).long()
                sigma_index_rot = torch.round(complex_t_rot.cpu() * (10 - 1)).long()
                sigma_index_tor = torch.round(complex_t_tor.cpu() * (10 - 1)).long()
                sigma_index_chi = torch.round(complex_t_chi.cpu() * (10 - 1)).long()
                meter_all.add(
                    [loss.cpu().detach(), tr_loss, rot_loss, tor_loss, chi_loss, tr_base_loss, rot_base_loss, tor_base_loss, chi_base_loss],
                    [sigma_index_tr, sigma_index_tr, sigma_index_rot, sigma_index_tor, sigma_index_chi, sigma_index_tr, sigma_index_rot,
                        sigma_index_tor, sigma_index_chi])

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    out = meter.summary()
    # if test_sigma_intervals > 0: out.update(meter_all.summary())
    return out

def get_pocket_metric(pred_pos, true_protein, metric):
    # assert pred_pos.shape == true_pos.shape
    # pred_pos = pred_protein.node_position
    true_pos = true_protein.node_position
    protein = true_protein

    pred_pos_per_residue = torch.zeros(protein.num_residue, len(atom_name_vocab), 3, device=true_pos.device)
    true_pos_per_residue = torch.zeros(protein.num_residue, len(atom_name_vocab), 3, device=true_pos.device)
    pred_pos_per_residue[protein.atom2residue, protein.atom_name] = pred_pos
    true_pos_per_residue[protein.atom2residue, protein.atom_name] = true_pos
    symm_true_pos_per_residue = _get_symm_atoms(true_pos_per_residue, protein.residue_type)

    # Symmetric alignment
    sc_rmsd_per_residue = _rmsd_per_residue(pred_pos_per_residue, true_pos_per_residue, protein.sidechain37_mask)
    sc_sym_rmsd_per_residue = _rmsd_per_residue(pred_pos_per_residue, symm_true_pos_per_residue,
                                                protein.sidechain37_mask)
    rmsd_per_residue = _rmsd_per_residue(pred_pos_per_residue, true_pos_per_residue, protein.atom37_mask)
    sym_rmsd_per_residue = _rmsd_per_residue(pred_pos_per_residue, symm_true_pos_per_residue, protein.atom37_mask)
    
    sym_replace_mask = sc_rmsd_per_residue > sc_sym_rmsd_per_residue
    sc_rmsd_per_residue[sym_replace_mask] = sc_sym_rmsd_per_residue[sym_replace_mask]
    rmsd_per_residue[sym_replace_mask] = sym_rmsd_per_residue[sym_replace_mask]
    metric["sc_atom_rmsd_per_residue"] = sc_rmsd_per_residue
    metric['atom_rmsd_per_residue'] = rmsd_per_residue

    true_pos_per_residue[sym_replace_mask] = symm_true_pos_per_residue[sym_replace_mask]
    true_pos = true_pos_per_residue[protein.atom2residue, protein.atom_name]
    pred_chi = rotamer.get_chis(protein, pred_pos)
    true_chi = rotamer.get_chis(protein, true_pos)
    chi_diff = (pred_chi - true_chi).abs()
    chi_ae = torch.minimum(chi_diff, 2 * np.pi - chi_diff)
    chi_ae_periodic = torch.minimum(chi_ae, np.pi - chi_ae)
    chi_ae[protein.chi_1pi_periodic_mask] = chi_ae_periodic[protein.chi_1pi_periodic_mask]
    # metric["chi_ae_rad"] = chi_ae[protein.chi_mask]  # [num_residue, 4]
    metric["chi_ae_deg"] = chi_ae[protein.chi_mask] * 180 / np.pi  # [num_residue, 4]
    for i in range(NUM_CHI_ANGLES):
        # metric[f"chi_{i}_ae_rad"] = chi_ae[:, i][protein.chi_mask[:, i]]
        if protein.chi_mask[:, i].sum() == 0:
            metric[f"chi_{i}_ae_deg"] = chi_ae.new_tensor([0.0])
        else:
            metric[f"chi_{i}_ae_deg"] = chi_ae[:, i][protein.chi_mask[:, i]] * 180 / np.pi

    return metric

def inference_epoch(model, complex_graphs, device, t_to_sigma, args):
    t_schedule = get_t_schedule(inference_steps=args.inference_steps)
    tr_schedule, rot_schedule, tor_schedule, chi_schedule = t_schedule, t_schedule, t_schedule, t_schedule

    dataset = ListDataset(complex_graphs)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False) # Just one by one here, keep in mind in reading next codes, hard to change
    rmsds = []
    chi_metric = {}

    for orig_complex_graph in tqdm(loader):
        data_list = [copy.deepcopy(orig_complex_graph)]
        randomize_position(data_list, args.no_torsion, False, 
                           args.no_chi_angle, args.tr_sigma_max, 
                           args.atom_radius, args.atom_max_neighbors)

        predictions_list = None
        failed_convergence_counter = 0
        while predictions_list == None:
            try:
                predictions_list, confidences = sampling(data_list=data_list, model=model.module if device.type=='cuda' else model,
                                                         inference_steps=args.inference_steps,
                                                         tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                         tor_schedule=tor_schedule, chi_schedule=chi_schedule,
                                                         device=device, t_to_sigma=t_to_sigma, model_args=args)
            except Exception as e:
                if 'failed to converge' in str(e):
                    failed_convergence_counter += 1
                    if failed_convergence_counter > 5:
                        print('| WARNING: SVD failed to converge 5 times - skipping the complex')
                        break
                    print('| WARNING: SVD failed to converge - trying again with a new sample')
                else:
                    raise e
        if failed_convergence_counter > 5: continue
        if args.no_torsion:
            orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() +
                                                     orig_complex_graph.original_center.cpu().numpy())

        filterHs = torch.not_equal(predictions_list[0]['ligand'].x[:, 0], 0).cpu().numpy()

        if isinstance(orig_complex_graph['ligand'].orig_pos, list):
            orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]

        ligand_pos = np.asarray(
            [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list])
        orig_ligand_pos = np.expand_dims(
            orig_complex_graph['ligand'].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy(), axis=0)
        rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))
        rmsds.append(rmsd)

        pred_protein = predictions_list[0]['sidechain']
        true_protein = orig_complex_graph['sidechain']
        chi_ae = get_pocket_metric(pred_protein.node_position, true_protein, {})
        for k, v in chi_ae.items():
            if k not in chi_metric:
                chi_metric[k] = []
            chi_metric[k].append(v.mean())


    # pdb.set_trace()
    rmsds = np.array(rmsds)
    for k, v in chi_metric.items():
        chi_metric[k] = np.array(v).mean()
    losses = {'rmsds_lt2': (100 * (rmsds < 2).sum() / len(rmsds)),
              'rmsds_lt5': (100 * (rmsds < 5).sum() / len(rmsds))}
    losses.update(chi_metric)
    return losses
