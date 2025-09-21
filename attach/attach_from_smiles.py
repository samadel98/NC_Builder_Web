#!/usr/bin/env python3
"""
attach_from_smiles.py — Multi-ligand, multi-pass QD passivation from SMILES.

This script allows for sequential passivation of different dummy types with
mixtures of ligands, providing control over ratios and spatial distribution.

Example for a two-pass passivation:
  1. Passivate Cl sites with a 25/75 segmented mix of two anionic ligands.
  2. Passivate Rb sites on the resulting structure with a 40/60 random mix
     of two cationic ligands.

python attach_from_smiles.py \
    --qd initial_dot.xyz \
    --out_prefix final_passivated_dot \
    --job-ligands "CCCC(=O)O" "CCCCS" \
    --job-dummy Cl \
    --job-dist 0.25:0.75:segmented \
    --job-ligands "CN" "CCCN" \
    --job-dummy Rb \
    --job-dist 0.40:0.60:random
"""

import argparse
import math
import random
from typing import Dict, List, Tuple

import numpy as np
from ase import Atoms, io
from scipy.spatial import KDTree
# Import for advanced distribution algorithm
from scipy.spatial.distance import pdist, squareform


# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

# FG helpers
from functional_groups_class import (
    get_fg_registry,
    detect_fg_matches_neutral,
    rdconf_to_numpy,
)

# ——— Core Math & Geometry Helpers (unchanged) ——————————————————————
def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n else v

def rotation_matrix_from_vectors(a, b):
    a, b = unit(a), unit(b)
    v = np.cross(a, b); s = np.linalg.norm(v)
    if s < 1e-12: return np.eye(3) if np.dot(a, b) > 0 else -np.eye(3)
    c = np.dot(a, b)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))

def rotate_about_axis(coords, axis, ang):
    axis = unit(axis)
    c, s = math.cos(ang), math.sin(ang)
    ux, uy, uz = axis
    R = np.array([
        [c+ux*ux*(1-c), ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
        [uy*ux*(1-c)+uz*s, c+uy*uy*(1-c), uy*uz*(1-c)-ux*s],
        [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s, c+uz*uz*(1-c)]])
    return (R @ coords.T).T

def local_normal_hybrid(qd, qd_pos, dummy_idx, dummy_symbol, k=12):
    core_indices = [i for i, a in enumerate(qd) if a.symbol != dummy_symbol]
    if not core_indices: return radial_normal(qd, qd_pos, dummy_idx)
    core_pos = qd_pos[core_indices]
    kdt = KDTree(core_pos)
    dists, neigh_indices = kdt.query(qd_pos[dummy_idx], k=min(k, len(core_pos)))
    neigh_pts = core_pos[neigh_indices]
    weights = np.exp(-(dists / (np.median(dists) + 1e-9))**2)
    center = np.average(neigh_pts, axis=0, weights=weights)
    X = neigh_pts - center
    cov = (X.T * weights) @ X
    _, V = np.linalg.eigh(cov)
    n_pca = V[:, 0]
    n_rad = radial_normal(qd, qd_pos, dummy_idx)
    n_h = 0.7 * n_pca + 0.3 * n_rad
    n_h = unit(n_h)
    return -n_h if np.dot(n_h, n_rad) < 0 else n_h

def radial_normal(qd, qd_pos, dummy_idx):
    return unit(qd_pos[dummy_idx] - qd.get_center_of_mass())

def outward_normal(qd, idx, n):
    com = qd.get_center_of_mass()
    to_anchor = qd[idx].position - com
    return -n if np.dot(to_anchor, n) < 0 else n

# ——— RDKit & Ligand Preparation (unchanged) ———————————————————————
def smiles_to_3d_mol(smiles, seed=1337, ff="uff"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError(f"Cannot parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) < 0:
        raise RuntimeError(f"Embedding failed for SMILES: {smiles}")
    if ff == "mmff" and AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    return mol

def compute_anchor_frame(mol, donor_idx):
    coords = rdconf_to_numpy(mol)
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    anchor_center = coords[donor_idx]
    heavy_indices = [i for i, n in enumerate(numbers) if n > 1 and i != donor_idx]
    if not heavy_indices:
        neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(donor_idx).GetNeighbors()]
        vec = coords[neighbors[0]] - anchor_center if neighbors else np.array([0., 0., 1.])
    else:
        vec = coords[heavy_indices].mean(axis=0) - anchor_center
    return anchor_center, unit(vec)

# ——— Sterics & Placement ——————————————————————————————
pt = Chem.GetPeriodicTable()
vdw_radii = {1: 1.2, 6: 1.7, 7: 1.55, 8: 1.52, 15: 1.8, 16: 1.8, **{i: pt.GetRvdw(i) for i in range(1, 119)}}
def get_vdw(Z): return vdw_radii.get(Z, 2.0)

def build_sterics_tree(atoms, exclude_symbols, mode):
    coords, radii = [], []
    for a in atoms:
        if a.symbol in exclude_symbols or (mode == "heavy" and a.number == 1):
            continue
        coords.append(a.position)
        radii.append(get_vdw(a.number))
    if not coords: return KDTree([np.zeros(3)]), np.array([2.0])
    return KDTree(coords), np.asarray(radii, float)

def _refine_one_ligand_vectorized(site_config, environment, args):
    """
    Highly optimized and vectorized version of the refinement function.
    """
    numbers, lig_coords, anchor_center, anchor_vec = (
        site_config[k] for k in ["numbers", "coords", "anchor_center", "anchor_vec"])
    n0, dummy_pos, metal_pos, bond_len = (
        site_config[k] for k in ["n0", "dpos", "metal_pos", "bond_len"])
    other_dummies = site_config["other_dummy_positions"]
    dummy_symbol = site_config["dummy_symbol"]

    # Initial alignment (once per ligand)
    lig_coords0 = lig_coords - anchor_center
    aligned = (rotation_matrix_from_vectors(anchor_vec, n0) @ lig_coords0.T).T

    tree, radii = build_sterics_tree(environment, {dummy_symbol}, args.sterics_mode)
    cand_idx = np.arange(len(numbers)) if args.sterics_mode != "heavy" else np.where(numbers > 1)[0]
    
    base_extra = float(bond_len) - np.linalg.norm(metal_pos - dummy_pos) if args.anchor_mode == "dummy" else 0.0
    
    # --- VECTORIZED SCAN ---
    phis = np.deg2rad(np.arange(0, 360, args.coarse_step_deg))
    extras = args.offset_out + np.arange(args.adaptive_offset_steps + 1) * args.adaptive_offset_step
    phi_grid, extra_grid = np.meshgrid(phis, extras)
    phi_flat, extra_flat = phi_grid.flatten(), extra_grid.flatten()
    num_poses = len(phi_flat)

    c, s = np.cos(phi_flat), np.sin(phi_flat)
    ux, uy, uz = n0
    R_flat = np.array([
        [c + ux*ux*(1-c),   ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c),   uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)   ]
    ])
    R = R_flat.transpose(2, 0, 1)

    all_rotated = np.einsum('pij,aj->pai', R, aligned)
    anchor_points = dummy_pos + n0 * (base_extra + extra_flat[:, np.newaxis])
    all_coords = all_rotated + anchor_points[:, np.newaxis, :]

    query_points = all_coords[:, cand_idx, :].reshape(-1, 3)
    dist_vals, neighbor_indices = tree.query(query_points, k=1)
    
    dist_vals = dist_vals.reshape(num_poses, -1)
    if args.sterics_mode == "vdw":
        cand_radii = np.array([get_vdw(Z) for Z in numbers[cand_idx]])
        env_radii_matched = radii[neighbor_indices].reshape(num_poses, -1)
        clearances = dist_vals - (cand_radii[np.newaxis, :] + env_radii_matched)
    else:
        clearances = dist_vals
    steric_scores = np.min(clearances, axis=1)

    if other_dummies and args.neighbor_repulsion > 0:
        neighbor_dirs = np.array([unit(p - dummy_pos) for p in other_dummies])
        lig_coms = np.mean(all_coords, axis=1)
        tail_vecs = unit(lig_coms - anchor_points)
        alignments = np.einsum('pi,ki->pk', tail_vecs, neighbor_dirs)
        alignments[alignments < 0] = 0
        penalties = args.neighbor_repulsion * np.sum(alignments**2, axis=1)
    else:
        penalties = np.zeros(num_poses)

    final_scores = steric_scores - penalties
    best_pose_idx = np.argmax(final_scores)
    best_coords = all_coords[best_pose_idx]

    final_clearance = steric_scores[best_pose_idx]
    metric_str = "min-vdW-clearance" if args.sterics_mode == "vdw" else "min-dist"
    print(f"      → Final clearance: {final_clearance:.3f} Å ({metric_str})")

    return Atoms(numbers=list(numbers), positions=best_coords)

# ——— Ligand Distribution Algorithm —————————————————————
def _get_spatially_ordered_indices(dummy_pos, maximize_dist=True):
    """
    Return an ordering of site indices. Robust to non-numeric lists.
    If maximize_dist=True: farthest-pair seed + greedy farthest-next.
    Else: sweep along principal variance axis.
    """
    try:
        P = np.asarray(dummy_pos, dtype=float)
        if P.ndim != 2 or P.shape[1] != 3:
            # force [ [x,y,z], ... ] float
            P = np.array([[float(x), float(y), float(z)] for (x, y, z) in dummy_pos], dtype=float)

        n = len(P)
        if n <= 1:
            return list(range(n))

        D = squareform(pdist(P, metric="euclidean"))
        if not np.all(np.isfinite(D)):
            return list(range(n))

        if not maximize_dist:
            axis = np.argmax(np.var(P, axis=0))
            return list(np.argsort(P[:, axis]).astype(int))

        i, j = np.unravel_index(np.argmax(D), D.shape)
        order = [int(i), int(j)]
        remaining = set(range(n)) - set(order)

        while remaining:
            k = max(remaining, key=lambda idx: min(D[idx, order]))
            order.append(int(k))
            remaining.remove(k)
        return order
    except Exception:
        return list(range(len(dummy_pos)))

def create_site_assignments(dummy_indices, dummy_pos, ratios, strategy):
    num_sites = len(dummy_indices)
    counts = [int(round(r * num_sites)) for r in ratios]
    counts[-1] = num_sites - sum(counts[:-1])
    assignments = [i for i, count in enumerate(counts) for _ in range(count)]
    dummy_pos = np.asarray(dummy_pos, dtype=float)
    spatially_ordered_indices = _get_spatially_ordered_indices(
      dummy_pos, maximize_dist=(strategy == 'uniform')
    )
    if strategy == 'random':
        random.shuffle(assignments)
        return assignments
    spatially_ordered_indices = _get_spatially_ordered_indices(dummy_pos, maximize_dist=(strategy == 'uniform'))
    final_assignments = [0] * num_sites
    for i, site_idx in enumerate(spatially_ordered_indices):
        final_assignments[site_idx] = assignments[i]
    return final_assignments

def execute_passivation_job(qd: Atoms, job: dict, args: argparse.Namespace) -> Atoms:
    print(f"\n--- Starting Passivation Job ---")
    print(f"  Role: {job['role']}, Dummy: '{job['dummy']}'")
    print(f"  Ligands: {job['smiles']}")
    print(f"  Distribution: {job['ratios']} ({job['strategy']})")

    dummies = [i for i, a in enumerate(qd) if a.symbol == job['dummy']]
    if not dummies:
        print(f"  Warning: No dummy atoms with symbol '{job['dummy']}' found. Skipping job.")
        return qd
    print(f"  Found {len(dummies)} sites to passivate.")

    qd_pos, dummy_pos = qd.get_positions(), qd.get_positions()[dummies]
    site_assignments = create_site_assignments(dummies, dummy_pos, job['ratios'], job['strategy'])
    
    print("  Site assignments (ligand_idx @ dummy_idx):")
    for i, didx in enumerate(dummies):
        print(f"    - Site {didx}: Ligand {site_assignments[i]} ('{job['smiles'][site_assignments[i]]}')")

    print("  Pre-computing site configurations...")
    site_configs = []
    for i, didx in enumerate(dummies):
        # --- FIX: Added print statement for progress tracking ---
        print(f"    - Configuring site {didx} ({i+1}/{len(dummies)})...")
        ligand_idx = site_assignments[i]
        precomputed = job['precomputed_ligands'][ligand_idx]
        mol_site, fg_active, match_active = precomputed['mol'], precomputed['fg'], precomputed['match']
        
        at_nums, coords = np.array([a.GetAtomicNum() for a in mol_site.GetAtoms()]), rdconf_to_numpy(mol_site)
        donor_idx = donor_indices_for_fg(fg_active.name, mol_site, match_active)[0]
        _, metal_sym, metal_pos = nearest_metal_to_dummy(qd, didx, job['dummy'])
        
        site_configs.append({
            "didx": didx, "dpos": qd_pos[didx], 
            "n0": outward_normal(qd, didx, local_normal_hybrid(qd, qd_pos, didx, job['dummy'])),
            "numbers": at_nums, "coords": coords,
            "anchor_center": coords[donor_idx], "anchor_vec": compute_anchor_frame(mol_site, donor_idx)[1],
            "metal_pos": metal_pos, "bond_len": target_bond_length(metal_sym, at_nums[donor_idx], args.bond_len_override),
            "other_dummy_positions": [p for j, p in enumerate(dummy_pos) if dummies[j] != didx],
            "dummy_symbol": job['dummy']
        })

    placed_ligands = [Atoms() for _ in dummies]
    for pass_num in range(args.refinement_passes):
        print(f"  Refinement Pass {pass_num + 1}/{args.refinement_passes}...")
        next_pass_ligands = [Atoms() for _ in dummies]
        for i in range(len(dummies)):
            smiles = job['smiles'][site_assignments[i]]
            print(f"    Refining ligand {i+1}/{len(dummies)} ('{smiles}' at site {site_configs[i]['didx']})...")
            env_for_ligand = qd.copy()
            for j in range(len(dummies)):
                if i != j: env_for_ligand += placed_ligands[j]
            next_pass_ligands[i] = _refine_one_ligand_vectorized(site_configs[i], env_for_ligand, args)
        placed_ligands = next_pass_ligands

    core = qd.copy()
    del core[[i for i, a in enumerate(core) if a.symbol == job['dummy']]]
    return core + sum(placed_ligands, Atoms())

# ——— Utilities ————————————————
def build_per_site_variant(fg_instances, base_mol, chosen_idx, ff):
    fg_chosen, match_chosen = fg_instances[chosen_idx]
    m = Chem.Mol(base_mol)
    for j, (fg_obj, match) in enumerate(fg_instances):
        if j != chosen_idx: m = fg_obj.to_neutral(m, match)
    m = fg_chosen.to_ionic(m, match_chosen)
    try:
        if ff == "mmff" and AllChem.MMFFHasAllMoleculeParams(m): AllChem.MMFFOptimizeMolecule(m)
        else: AllChem.UFFOptimizeMolecule(m)
    except Exception: pass
    return m, (fg_chosen, match_chosen)

def donor_indices_for_fg(fg_name, mol, match):
    fg = fg_name.lower()
    if fg in ["thiolate", "ammonium"]: return [match[0]]
    if fg == "carboxylate": return [nb.GetIdx() for nb in mol.GetAtomWithIdx(match[0]).GetNeighbors() if nb.GetAtomicNum() == 8]
    return [match[0]]

DEFAULT_MX = {("Cd", "O"): 2.25, ("Cd", "S"): 2.55, ("Cd", "N"): 2.30, ("Pb", "O"): 2.50, ("Pb", "S"): 2.80}
def target_bond_length(metal, donor_Z, override):
    if override: return float(override)
    donor = {8: "O", 16: "S", 7: "N"}.get(donor_Z)
    return DEFAULT_MX.get((metal, donor), 2.40)

def nearest_metal_to_dummy(qd, dummy_idx, exclude):
    dpos = qd.positions[dummy_idx]
    dists = [(np.linalg.norm(a.position - dpos), i, a.symbol, a.position) for i, a in enumerate(qd) if a.symbol != exclude]
    if not dists: raise RuntimeError("Could not find nearest metal.")
    return min(dists)[1:]

# ——— Main Controller —————————————————————————————————————————————
def main():
    ap = argparse.ArgumentParser(description="Multi-ligand, multi-pass QD passivation from SMILES.")
    ap.add_argument("--qd", required=True, help="Input QD XYZ file with dummy atoms.")
    ap.add_argument("--out_prefix", required=True, help="Prefix for output XYZ file.")
    ap.add_argument('--job-ligands', action='append', nargs='+', help='SMILES strings for a passivation job.')
    ap.add_argument('--job-dummy', action='append', help='Dummy symbol for a passivation job.')
    ap.add_argument('--job-dist', action='append', help='Distribution string (e.g., "0.25:0.75:random").')
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--ff", choices=["uff", "mmff"], default="uff")
    ap.add_argument("--refinement_passes", type=int, default=3, help="Number of iterative refinement passes (default: 3).")
    ap.add_argument("--neigh", type=int, default=8)
    ap.add_argument("--coarse_step_deg", type=float, default=20.0)
    ap.add_argument("--sterics_mode", choices=["heavy", "all", "vdw"], default="vdw")
    ap.add_argument("--sterics_margin", type=float, default=0.25)
    ap.add_argument("--warn_tol", type=float, default=1.4)
    ap.add_argument("--adaptive_offset_steps", type=int, default=4)
    ap.add_argument("--adaptive_offset_step", type=float, default=0.15)
    ap.add_argument("--anchor_mode", choices=["dummy", "metal"], default="dummy")
    ap.add_argument("--offset_out", type=float, default=0.0)
    ap.add_argument("--bond_len_override", type=float, default=None)
    ap.add_argument("--neighbor_repulsion", type=float, default=0.5)

    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    num_jobs = len(args.job_ligands) if args.job_ligands else 0
    if num_jobs != len(args.job_dummy) or num_jobs != len(args.job_dist):
        raise ValueError("Mismatch in number of --job-ligands, --job-dummy, and --job-dist arguments.")

    jobs = []
    reg = get_fg_registry()
    print("--- Pre-processing and optimizing unique ligand types ---")
    for i in range(num_jobs):
        smiles_list, dist_parts = args.job_ligands[i], args.job_dist[i].split(':')
        strategy, ratios = dist_parts[-1], [float(r) for r in dist_parts[:-1]]
        
        if not math.isclose(sum(ratios), 1.0): raise ValueError(f"Ratios for job {i+1} do not sum to 1.0.")
        if len(ratios) != len(smiles_list): raise ValueError(f"Ratio/ligand mismatch for job {i+1}.")
        if strategy not in ['random', 'segmented', 'uniform']: raise ValueError(f"Invalid strategy '{strategy}' for job {i+1}.")

        precomputed_ligands = []
        role = None
        for smiles in smiles_list:
            print(f"  Processing SMILES: '{smiles}'")
            mol_neutral = smiles_to_3d_mol(smiles, args.seed, args.ff)
            fg_matches = detect_fg_matches_neutral(mol_neutral, 'anion', reg) or detect_fg_matches_neutral(mol_neutral, 'cation', reg)
            if not fg_matches: raise ValueError(f"No recognizable functional group for SMILES: {smiles}")
            if role is None: role = fg_matches[0][0].role

            print(f"    → Optimizing ionic form...")
            ionic_mol, (fg_active, match_active) = build_per_site_variant(fg_matches, mol_neutral, 0, args.ff)
            precomputed_ligands.append({
                "mol": ionic_mol, "fg": fg_active, "match": match_active
            })

        jobs.append({
            "smiles": smiles_list, "dummy": args.job_dummy[i], "role": role,
            "ratios": ratios, "strategy": strategy, "precomputed_ligands": precomputed_ligands
        })

    current_qd = io.read(args.qd)
    for i, job in enumerate(jobs):
        current_qd = execute_passivation_job(current_qd, job, args)
        io.write(f"{args.out_prefix}_pass_{i+1}_{job['dummy']}.xyz", current_qd)
    
    final_fname = f"{args.out_prefix}_final.xyz"
    io.write(final_fname, current_qd)
    print(f"\n--- All jobs complete. Final structure written to: {final_fname} ---")

if __name__ == "__main__":
    main()


