import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.stats import beta
import pdb

from utils.geometry import axis_angle_to_matrix, rigid_transform_Kabsch_3D_torch
from utils.torsion import modify_conformer_torsion_angles


def t_to_sigma(t_tr, t_rot, t_tor, t_chi, args):
    # if linear_tr_schedule:
    #     tr_sigma = t_tr # for small step size near pocket center, it's buggy
    # else:
    tr_sigma = args.tr_sigma_min ** (1-t_tr) * args.tr_sigma_max ** t_tr
    rot_sigma = args.rot_sigma_min ** (1-t_rot) * args.rot_sigma_max ** t_rot
    tor_sigma = args.tor_sigma_min ** (1-t_tor) * args.tor_sigma_max ** t_tor
    if getattr(args, 'chi_sigma_min', None) is not None:
        chi_sigma = args.chi_sigma_min ** (1-t_chi) * args.chi_sigma_max ** t_chi #This noise_schedule is incossistent with the training process
        # Actually reverse noise schedule just affects the step size of denoising, that's why incossistent noise schedule doesn't affect the results
        # chi_sigma = t_chi #Start from randomized chi angles but use a relatively small noise schedule, the same with the training process, check its effect on the results
    else:
        chi_sigma = None
    return tr_sigma, rot_sigma, tor_sigma, chi_sigma


def modify_conformer(data, tr_update, rot_update, torsion_updates):
    lig_center = torch.mean(data['ligand'].pos, dim=0, keepdim=True)
    rot_mat = axis_angle_to_matrix(rot_update.squeeze())
    if (torch.abs(tr_update) < 100).any(): # ensure the update is normal
        rigid_new_pos = (data['ligand'].pos - lig_center) @ rot_mat.T + tr_update + lig_center
    else:
        # print('tr_update or rot_update is abnormal, use the original conformer instead.Will try again in next SDE discrete step.')
        rigid_new_pos = data['ligand'].pos # keep the original conformer and only conduct torsion updates
        data.success = torch.tensor([False])
    
    if torsion_updates is not None:
        try:
            flexible_new_pos = modify_conformer_torsion_angles(rigid_new_pos,
                                                            data['ligand', 'ligand'].edge_index.T[data['ligand'].edge_mask],
                                                            data['ligand'].mask_rotate if isinstance(data['ligand'].mask_rotate, np.ndarray) else data['ligand'].mask_rotate[0],
                                                            torsion_updates).to(rigid_new_pos.device)
            R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
            aligned_flexible_pos = flexible_new_pos @ R.T + t.T
            data['ligand'].pos = aligned_flexible_pos
        except Exception as e:
            # The error is caused by the tr_update and rot_update, the irregular large number in the update, which comes from the inconsistent network output
            # data['ligand'].pos = rigid_new_pos
            print('modify_conformer_torsion_angles failed, use the original conformer instead.Will try again in next SDE discrete step:', e)
            # determine whether to accept the conformer in the save procudure
            # set the value of data.success to tensor([False])
            data.success = torch.tensor([False])
            # pdb.set_trace()
    else:
        data['ligand'].pos = rigid_new_pos
    return data


def sinusoidal_embedding(timesteps, embedding_dim, max_positions=10000):
    """ from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py   """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size//2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return emb


def get_timestep_embedding(embedding_type, embedding_dim, embedding_scale=10000):
    if embedding_type == 'sinusoidal':
        emb_func = (lambda x : sinusoidal_embedding(embedding_scale * x, embedding_dim))
    elif embedding_type == 'fourier':
        emb_func = GaussianFourierProjection(embedding_size=embedding_dim, scale=embedding_scale)
    else:
        raise NotImplemented
    return emb_func


def get_t_schedule(inference_steps):
    # return np.linspace(1, 0, inference_steps + 1)[:-1]
    return np.linspace(0.7, 0, inference_steps + 1)[:-1] # Trunck time for pocket center initialization


def set_time(complex_graphs, t_tr, t_rot, t_tor, t_chi, batchsize, all_atoms, device):
    complex_graphs['ligand'].node_t = {
        'tr': t_tr * torch.ones(complex_graphs['ligand'].num_nodes).to(device),
        'rot': t_rot * torch.ones(complex_graphs['ligand'].num_nodes).to(device),
        'tor': t_tor * torch.ones(complex_graphs['ligand'].num_nodes).to(device),
        'chi': t_chi * torch.ones(complex_graphs['ligand'].num_nodes).to(device)}
    complex_graphs['receptor'].node_t = {
        'tr': t_tr * torch.ones(complex_graphs['receptor'].num_nodes).to(device),
        'rot': t_rot * torch.ones(complex_graphs['receptor'].num_nodes).to(device),
        'tor': t_tor * torch.ones(complex_graphs['receptor'].num_nodes).to(device),
        'chi': t_chi * torch.ones(complex_graphs['receptor'].num_nodes).to(device)}
    complex_graphs.complex_t = {'tr': t_tr * torch.ones(batchsize).to(device),
                               'rot': t_rot * torch.ones(batchsize).to(device),
                               'tor': t_tor * torch.ones(batchsize).to(device),
                               'chi': t_chi * torch.ones(batchsize).to(device)}
    if all_atoms:
        complex_graphs['atom'].node_t = {
            'tr': t_tr * torch.ones(complex_graphs['atom'].num_nodes).to(device),
            'rot': t_rot * torch.ones(complex_graphs['atom'].num_nodes).to(device),
            'tor': t_tor * torch.ones(complex_graphs['atom'].num_nodes).to(device),
            'chi': t_chi * torch.ones(complex_graphs['atom'].num_nodes).to(device)}