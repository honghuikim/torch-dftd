from typing import Optional, Tuple

import numpy as np
import torch
from ase.neighborlist import primitive_neighbor_list
from ase.units import Bohr
from pymatgen.core import Structure
from torch import Tensor

import time

def calc_neighbor_by_ase(
    pos: Tensor, cell: Tensor, pbc: Tensor, cutoff: float
) -> Tuple[Tensor, Tensor]:
    idx_i, idx_j, S = primitive_neighbor_list(
        "ijS",
        pbc.detach().cpu().numpy(),
        cell.detach().cpu().numpy(),
        pos.detach().cpu().numpy(),
        cutoff,
    )
    edge_index = torch.tensor(np.stack([idx_i, idx_j], axis=0), device=pos.device)
    # convert int64 -> pos.dtype (float)
    S = torch.tensor(S, dtype=pos.dtype, device=pos.device)
    return edge_index, S


def calc_neighbor_by_pymatgen(
    pos: Tensor, cell: Tensor, pbc: Tensor, cutoff: float, indices: Optional[list] = None, gas_mof_only: bool = False,
) -> Tuple[Tensor, Tensor]:
    """calculate neighbor nodes in pbc condition.

    Implementation referred from https://github.com/Open-Catalyst-Project/ocp/blob/a5634ee4f0dc4a874752ab8d3117492ce83261ac/ocpmodels/preprocessing/atoms_to_graphs.py#L76
    under MIT license.

    Args:
        pos (Tensor):
        cell (Tensor):
        pbc (Tensor): periodic boundary condition.
        cutoff (float): cutoff distance to find neighbor
        indices (list): atom indices in interest, to calculate neighbor list.

    Returns:
        edge_index (Tensor): (2, n_edges) indices of edge, src -> dst.
        S (Tensor): (n_edges, 3) shift tensor
    """  # NOQA
    if not torch.all(pbc):
        raise NotImplementedError(f"pbc {pbc} must be true for all axis!")

    positions = pos.detach().cpu().numpy().copy()
    lattice = cell.detach().cpu().numpy().copy()
    n_atoms = positions.shape[0]
    symbols = np.ones(n_atoms)  # Dummy symbols to create `Structure`...

    struct = Structure(lattice, symbols, positions, coords_are_cartesian=True)
    if gas_mof_only:
        seq = [struct[i] for i in indices]
    else:
        seq = None
    c_index, n_index, offsets, n_distance = struct.get_neighbor_list(
        r=cutoff,
        sites=seq,
        numerical_tol=1e-8,
        exclude_self=True,
    )

    if seq is not None:
        #change unmatched indices to original indices
        for i, index in enumerate(indices):
            c_index[np.where(c_index == i)] = index
            #remove gas-gas self interaction
            indices_tmp = np.where(np.logical_and(np.logical_and(c_index==index, n_index==index), n_distance<1e-5))[0]
            c_index = np.delete(c_index, indices_tmp,0)
            n_index = np.delete(n_index, indices_tmp,0)
            offsets = np.delete(offsets, indices_tmp,0)
            n_distance = np.delete(n_distance, indices_tmp,0)
        #duplicate Framework-Gas Edges
        edges_frame_gas = np.where(~np.isin(n_index, indices))[0]
        c_index = np.concatenate((c_index, n_index[edges_frame_gas]), axis=0)
        n_index = np.concatenate((n_index, c_index[edges_frame_gas]), axis=0)
        offsets = np.concatenate((offsets, -offsets[edges_frame_gas]), axis=0)
        n_distance = np.concatenate((n_distance, n_distance[edges_frame_gas]), axis=0)
    '''
    if indices is not None:
        indices_interest = np.where(np.logical_or(np.isin(c_index, indices), np.isin(n_index, indices)))[0]
        indices_interest = torch.tensor(
            indices_interest, dtype=torch.long, device=pos.device
        )
    else:
        indices_interest = None
    '''
    indices_interest = None
    edge_index = torch.tensor(
        np.stack([c_index, n_index], axis=0), dtype=torch.long, device=pos.device
    )
    n_distance = torch.tensor(
        n_distance, dtype=torch.double, device=pos.device
    )
    S = torch.tensor(offsets, dtype=pos.dtype, device=pos.device)

    return edge_index, S, indices_interest, n_distance


def calc_edge_index(
    pos: Tensor,
    cell: Optional[Tensor] = None,
    pbc: Optional[Tensor] = None,
    indices: Optional[list] = None,
    cutoff: float = 95.0 * Bohr,
    bidirectional: bool = False,
    gas_mof_only: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Calculate atom pair as `edge_index`, and shift vector `S`.

    Args:
        pos (Tensor): atom positions in angstrom
        cell (Tensor): cell size in angstrom, None for non periodic system.
        pbc (Tensor): pbc condition, None for non periodic system.
        indices (list): atom indices in interest, to calculate neighbor list.
        cutoff (float): cutoff distance in angstrom
        bidirectional (bool): calculated `edge_index` is bidirectional or not.

    Returns:
        edge_index (Tensor): (2, n_edges)
        S (Tensor): (n_edges, 3) dtype is same with `pos`
    """
    if pbc is None or torch.all(~pbc):
        assert cell is None
        # Calculate distance brute force way
        distances = torch.sum((pos.unsqueeze(0) - pos.unsqueeze(1)).pow_(2), dim=2)
        right_ind, left_ind = torch.where(distances < cutoff**2)
        if bidirectional:
            edge_index = torch.stack(
                (left_ind[left_ind != right_ind], right_ind[left_ind != right_ind])
            )
        else:
            edge_index = torch.stack(
                (left_ind[left_ind < right_ind], right_ind[left_ind < right_ind])
            )
        n_edges = edge_index.shape[1]
        S = pos.new_zeros((n_edges, 3))
    else:
        if not bidirectional:
            raise NotImplementedError("bidirectional=False is not supported")
        if pos.shape[0] == 0:
            edge_index = torch.zeros([2, 0], dtype=torch.long, device=pos.device)
            S = torch.zeros_like(pos)
        else:
            try:
                start_time = time.time()
                edge_index, S, indices_interest, n_distance = calc_neighbor_by_pymatgen(pos, cell, pbc, cutoff, indices, gas_mof_only)
                end_time = time.time()
                print(f"pymatgen neighbor list search:{end_time-start_time}")
            except NotImplementedError:
                # This is slower.
                edge_index, S = calc_neighbor_by_ase(pos, cell, pbc, cutoff)

    return edge_index, S, indices_interest, n_distance
