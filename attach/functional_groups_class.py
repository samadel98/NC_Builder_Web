# functional_groups_class.py
# RDKit-based functional group detection, ionic transforms, and anchor frames
# for attaching ligands to nanocrystal surfaces.

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

# ——— Basic linear algebra helpers ———————————————————————————

def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n else v

def rdconf_to_numpy(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    N = mol.GetNumAtoms()
    xyz = np.zeros((N, 3))
    for i in range(N):
        p: Point3D = conf.GetAtomPosition(i)
        xyz[i] = [p.x, p.y, p.z]
    return xyz

def project_onto_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Project vector v onto plane with unit normal n."""
    n = unit(n)
    return v - np.dot(v, n) * n

# ——— Base definition ————————————————————————————————

@dataclass
class FGDef:
    """Functional group definition for QD attachment."""
    name: str           # e.g., "carboxylate"
    role: str           # 'anion' or 'cation'
    smarts: str         # primary SMARTS (info only; neutral/ionic patterns may be separate below)

    # Transform THIS match to ionic (binding) form
    def to_ionic(self, mol: Chem.Mol, match: Tuple[int, ...]) -> Chem.Mol:
        return mol

    # Transform THIS match to neutral (non-binding) form
    def to_neutral(self, mol: Chem.Mol, match: Tuple[int, ...]) -> Chem.Mol:
        return mol

    # Compute anchor center/vector from current conformer
    def anchor_center_and_vector(self, mol: Chem.Mol, match: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

# ——— Utilities for editing ———————————————————————————————

def remove_attached_h(edit_mol: Chem.EditableMol, mol_view: Chem.Mol, heavy_idx: int) -> Tuple[Chem.Mol, Optional[int]]:
    """Remove one H attached to heavy_idx if present; return new mol and removed H idx (or None)."""
    heavy = mol_view.GetAtomWithIdx(heavy_idx)
    H_idx = None
    for nb in heavy.GetNeighbors():
        if nb.GetAtomicNum() == 1:
            H_idx = nb.GetIdx()
            break
    if H_idx is not None:
        edit_mol.RemoveAtom(H_idx)
        mol_new = edit_mol.GetMol()
        Chem.SanitizeMol(mol_new)
        return mol_new, H_idx
    return mol_view, None

# ——— Carboxylate ————————————————————————————————————————

class CarboxylateFG(FGDef):
    """
    Carboxylic acid / carboxylate (anion role).
    Neutral SMARTS: [CX3](=O)[OX2H1]
    Ionic SMARTS:   [CX3](=O)[O-]

    Anchor:
      center = C (carbonyl carbon)
      vector = (C → alpha-carbon centroid), projected onto O–C–O plane.
    """
    smarts_neutral = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
    smarts_ionic   = Chem.MolFromSmarts("[CX3](=O)[O-]")

    def __init__(self):
        super().__init__("carboxylate", "anion", "[CX3](=O)[OX2H1],[CX3](=O)[O-]")

    def to_ionic(self, mol: Chem.Mol, match: Tuple[int, ...]) -> Chem.Mol:
        # If already ionic anywhere, keep; else convert THIS match (COOH -> COO-)
        if mol.GetSubstructMatches(self.smarts_ionic):
            return mol
        C_idx, O_c_idx, O_h_idx = match
        em = Chem.EditableMol(mol)
        mol, _ = remove_attached_h(em, mol, O_h_idx)  # deprotonate OH
        mol.GetAtomWithIdx(O_h_idx).SetFormalCharge(-1)
        mol.GetAtomWithIdx(O_c_idx).SetFormalCharge(0)
        Chem.SanitizeMol(mol)
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=80)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=80)
        except Exception:
            pass
        return mol

    def to_neutral(self, mol: Chem.Mol, match: Tuple[int, ...]) -> Chem.Mol:
        # If COO-, re-protonate to COOH (add H to O-)
        C_idx = match[0]
        C = mol.GetAtomWithIdx(C_idx)
        Ominus = None
        for nb in C.GetNeighbors():
            if nb.GetAtomicNum() == 8 and mol.GetAtomWithIdx(nb.GetIdx()).GetFormalCharge() == -1:
                Ominus = nb.GetIdx()
                break
        if Ominus is None:
            return mol
        em = Chem.EditableMol(mol)
        H_idx = em.AddAtom(Chem.Atom(1))
        mol = em.GetMol()
        conf = mol.GetConformer()
        opos = conf.GetAtomPosition(Ominus)
        conf.SetAtomPosition(H_idx, Point3D(opos.x + 0.95, opos.y, opos.z))
        mol = Chem.RWMol(mol)
        mol.AddBond(Ominus, H_idx, Chem.BondType.SINGLE)
        mol = mol.GetMol()
        mol.GetAtomWithIdx(Ominus).SetFormalCharge(0)
        Chem.SanitizeMol(mol)
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=60)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=60)
        except Exception:
            pass
        return mol

    def anchor_center_and_vector(self, mol: Chem.Mol, match: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        coords = rdconf_to_numpy(mol)
        C_idx = match[0]
        Cpos  = coords[C_idx]
        O_idxs = [nb.GetIdx() for nb in mol.GetAtomWithIdx(C_idx).GetNeighbors()
                  if mol.GetAtomWithIdx(nb.GetIdx()).GetAtomicNum() == 8]
        if len(O_idxs) < 2:
            all_O = [i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetAtomicNum() == 8]
            O_idxs = sorted(all_O, key=lambda oi: np.linalg.norm(coords[oi]-Cpos))[:2]
        O1, O2 = O_idxs[0], O_idxs[1]
        v1 = coords[O1] - Cpos
        v2 = coords[O2] - Cpos
        n_plane = unit(np.cross(v1, v2))
        nonO = [nb.GetIdx() for nb in mol.GetAtomWithIdx(C_idx).GetNeighbors()
                if mol.GetAtomWithIdx(nb.GetIdx()).GetAtomicNum() not in (1, 8)]
        if nonO:
            tail_centroid = coords[nonO].mean(axis=0)
            v_tail = tail_centroid - Cpos
        else:
            v_tail = unit(v1/np.linalg.norm(v1) + v2/np.linalg_norm(v2)) if np.linalg.norm(v1)>1e-8 and np.linalg.norm(v2)>1e-8 else v1
        v_anchor = project_onto_plane(v_tail, n_plane)
        if np.linalg.norm(v_anchor) < 1e-8:
            v_anchor = project_onto_plane(np.cross(n_plane, v1), n_plane)
        return Cpos, unit(v_anchor)

# ——— Phosphonate ————————————————————————————————————————————

class PhosphonateFG(FGDef):
    """
    Phosphonic acid / phosphonate (anion role).
    Neutral pattern (permissive): [PX4](=O)(O)(O)
    Transform: remove one P–O–H to make mono-anion.

    Anchor:
      center = P
      vector = (P → non-O neighbor centroid) projected onto plane spanned by two O neighbors.
    """
    neutral_pattern = Chem.MolFromSmarts("[PX4](=O)(O)(O)")

    def __init__(self):
        super().__init__("phosphonate", "anion", "[PX4](=O)(O)(O)")

    def to_ionic(self, mol: Chem.Mol, match: Tuple[int, ...]) -> Chem.Mol:
        P_idx = match[0]
        P = mol.GetAtomWithIdx(P_idx)
        OH_O_idx = None
        for nb in P.GetNeighbors():
            if nb.GetAtomicNum() == 8:
                oi = nb.GetIdx()
                if any(nbb.GetAtomicNum() == 1 for nbb in mol.GetAtomWithIdx(oi).GetNeighbors()):
                    OH_O_idx = oi
                    break
        if OH_O_idx is None:
            return mol
        em = Chem.EditableMol(mol)
        mol, _ = remove_attached_h(em, mol, OH_O_idx)
        mol.GetAtomWithIdx(OH_O_idx).SetFormalCharge(-1)
        Chem.SanitizeMol(mol)
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=80)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=80)
        except Exception:
            pass
        return mol

    def to_neutral(self, mol: Chem.Mol, match: Tuple[int, ...]) -> Chem.Mol:
        # If any O has -1, re-protonate to P–O–H
        P_idx = match[0]
        P = mol.GetAtomWithIdx(P_idx)
        Ominus = None
        for nb in P.GetNeighbors():
            if nb.GetAtomicNum() == 8 and mol.GetAtomWithIdx(nb.GetIdx()).GetFormalCharge() == -1:
                Ominus = nb.GetIdx()
                break
        if Ominus is None:
            return mol
        em = Chem.EditableMol(mol)
        H_idx = em.AddAtom(Chem.Atom(1))
        mol = em.GetMol()
        conf = mol.GetConformer()
        opos = conf.GetAtomPosition(Ominus)
        conf.SetAtomPosition(H_idx, Point3D(opos.x + 0.98, opos.y, opos.z))
        mol = Chem.RWMol(mol)
        mol.AddBond(Ominus, H_idx, Chem.BondType.SINGLE)
        mol = mol.GetMol()
        mol.GetAtomWithIdx(Ominus).SetFormalCharge(0)
        Chem.SanitizeMol(mol)
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=60)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=60)
        except Exception:
            pass
        return mol

    def anchor_center_and_vector(self, mol: Chem.Mol, match: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        coords = rdconf_to_numpy(mol)
        P_idx = match[0]
        Ppos  = coords[P_idx]
        O_nb, nonO_nb = [], []
        P = mol.GetAtomWithIdx(P_idx)
        for nb in P.GetNeighbors():
            if nb.GetAtomicNum() == 8: O_nb.append(nb.GetIdx())
            elif nb.GetAtomicNum() != 1: nonO_nb.append(nb.GetIdx())
        if len(O_nb) >= 2:
            v1 = coords[O_nb[0]] - Ppos
            v2 = coords[O_nb[1]] - Ppos
            n_plane = unit(np.cross(v1, v2))
        else:
            neighs = [nb.GetIdx() for nb in P.GetNeighbors()]
            if len(neighs) >= 2:
                v1 = coords[neighs[0]] - Ppos; v2 = coords[neighs[1]] - Ppos
                n_plane = unit(np.cross(v1, v2))
            else:
                n_plane = np.array([0.0, 0.0, 1.0])
        if nonO_nb:
            tail_centroid = coords[nonO_nb].mean(axis=0)
            v_tail = tail_centroid - Ppos
        else:
            vec = np.zeros(3)
            for oi in O_nb: vec += coords[oi] - Ppos
            v_tail = vec if np.linalg.norm(vec) > 1e-8 else np.array([1.0, 0.0, 0.0])
        v_anchor = project_onto_plane(v_tail, n_plane)
        if np.linalg.norm(v_anchor) < 1e-8:
            v_anchor = project_onto_plane(np.cross(n_plane, v1), n_plane)
        return Ppos, unit(v_anchor)

# ——— Sulfonate ————————————————————————————————————————————————

class SulfonateFG(FGDef):
    """
    Sulfonic acid / sulfonate (anion role).
    Neutral pattern: [#16X6](=O)(=O)O

    Anchor:
      center = S
      vector = (S → non-O neighbor centroid) projected onto O–S–O plane.
    """
    neutral_pattern = Chem.MolFromSmarts("[#16X6](=O)(=O)O")

    def __init__(self):
        super().__init__("sulfonate", "anion", "[#16X6](=O)(=O)O")

    def to_ionic(self, mol: Chem.Mol, match: Tuple[int, ...]) -> Chem.Mol:
        S_idx = match[0]
        S = mol.GetAtomWithIdx(S_idx)
        OH_O_idx = None
        for nb in S.GetNeighbors():
            if nb.GetAtomicNum() == 8:
                oi = nb.GetIdx()
                if any(nbb.GetAtomicNum() == 1 for nbb in mol.GetAtomWithIdx(oi).GetNeighbors()):
                    OH_O_idx = oi
                    break
        if OH_O_idx is None:
            return mol
        em = Chem.EditableMol(mol)
        mol, _ = remove_attached_h(em, mol, OH_O_idx)
        mol.GetAtomWithIdx(OH_O_idx).SetFormalCharge(-1)
        Chem.SanitizeMol(mol)
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=80)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=80)
        except Exception:
            pass
        return mol

    def to_neutral(self, mol: Chem.Mol, match: Tuple[int, ...]) -> Chem.Mol:
        S_idx = match[0]
        S = mol.GetAtomWithIdx(S_idx)
        Ominus = None
        for nb in S.GetNeighbors():
            if nb.GetAtomicNum() == 8 and mol.GetAtomWithIdx(nb.GetIdx()).GetFormalCharge() == -1:
                Ominus = nb.GetIdx()
                break
        if Ominus is None:
            return mol
        em = Chem.EditableMol(mol)
        H_idx = em.AddAtom(Chem.Atom(1))
        mol = em.GetMol()
        conf = mol.GetConformer()
        opos = conf.GetAtomPosition(Ominus)
        conf.SetAtomPosition(H_idx, Point3D(opos.x + 0.98, opos.y, opos.z))
        mol = Chem.RWMol(mol)
        mol.AddBond(Ominus, H_idx, Chem.BondType.SINGLE)
        mol = mol.GetMol()
        mol.GetAtomWithIdx(Ominus).SetFormalCharge(0)
        Chem.SanitizeMol(mol)
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=60)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=60)
        except Exception:
            pass
        return mol

    def anchor_center_and_vector(self, mol: Chem.Mol, match: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        coords = rdconf_to_numpy(mol)
        S_idx = match[0]
        Spos  = coords[S_idx]
        O_nb, nonO_nb = [], []
        S = mol.GetAtomWithIdx(S_idx)
        for nb in S.GetNeighbors():
            if nb.GetAtomicNum() == 8: O_nb.append(nb.GetIdx())
            elif nb.GetAtomicNum() != 1: nonO_nb.append(nb.GetIdx())
        if len(O_nb) >= 2:
            v1 = coords[O_nb[0]] - Spos
            v2 = coords[O_nb[1]] - Spos
            n_plane = unit(np.cross(v1, v2))
        else:
            neighs = [nb.GetIdx() for nb in S.GetNeighbors()]
            if len(neighs) >= 2:
                v1 = coords[neighs[0]] - Spos; v2 = coords[neighs[1]] - Spos
                n_plane = unit(np.cross(v1, v2))
            else:
                n_plane = np.array([0.0, 0.0, 1.0])
        if nonO_nb:
            tail_centroid = coords[nonO_nb].mean(axis=0)
            v_tail = tail_centroid - Spos
        else:
            vec = np.zeros(3)
            for oi in O_nb: vec += coords[oi] - Spos
            v_tail = vec if np.linalg.norm(vec) > 1e-8 else np.array([1.0, 0.0, 0.0])
        v_anchor = project_onto_plane(v_tail, n_plane)
        if np.linalg.norm(v_anchor) < 1e-8:
            v_anchor = project_onto_plane(np.cross(n_plane, v1), n_plane)
        return Spos, unit(v_anchor)

# ——— Thiolate ————————————————————————————————————————————————

class ThiolateFG(FGDef):
    """
    Thiol / thiolate (anion role).
    Neutral SMARTS: [SX2H1]

    Anchor:
      center = S
      vector = (S → alpha-carbon centroid) projected onto a local plane.
    """
    neutral_pattern = Chem.MolFromSmarts("[SX2H1]")

    def __init__(self):
        super().__init__("thiolate", "anion", "[SX2H1]")

    def to_ionic(self, mol: Chem.Mol, match: Tuple[int, ...]) -> Chem.Mol:
        S_idx = match[0]
        em = Chem.EditableMol(mol)
        mol, _ = remove_attached_h(em, mol, S_idx)
        mol.GetAtomWithIdx(S_idx).SetFormalCharge(-1)
        Chem.SanitizeMol(mol)
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=80)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=80)
        except Exception:
            pass
        return mol

    def to_neutral(self, mol: Chem.Mol, match: Tuple[int, ...]) -> Chem.Mol:
        S_idx = match[0]
        S = mol.GetAtomWithIdx(S_idx)
        if S.GetFormalCharge() < 0:
            em = Chem.EditableMol(mol)
            H_idx = em.AddAtom(Chem.Atom(1))
            mol = em.GetMol()
            conf = mol.GetConformer()
            spos = conf.GetAtomPosition(S_idx)
            conf.SetAtomPosition(H_idx, Point3D(spos.x + 1.0, spos.y, spos.z))
            mol = Chem.RWMol(mol)
            mol.AddBond(S_idx, H_idx, Chem.BondType.SINGLE)
            mol = mol.GetMol()
            mol.GetAtomWithIdx(S_idx).SetFormalCharge(0)
            Chem.SanitizeMol(mol)
            try:
                if AllChem.MMFFHasAllMoleculeParams(mol):
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=60)
                else:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=60)
            except Exception:
                pass
        return mol

    def anchor_center_and_vector(self, mol: Chem.Mol, match: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        coords = rdconf_to_numpy(mol)
        S_idx = match[0]
        Spos  = coords[S_idx]
        neighs = [nb.GetIdx() for nb in mol.GetAtomWithIdx(S_idx).GetNeighbors()]
        if len(neighs) >= 2:
            v1 = coords[neighs[0]] - Spos; v2 = coords[neighs[1]] - Spos
            n_plane = unit(np.cross(v1, v2))
        else:
            n_plane = np.array([0.0, 0.0, 1.0])
        carbons = [i for i in neighs if mol.GetAtomWithIdx(i).GetAtomicNum() == 6]
        if carbons:
            ccent = coords[carbons].mean(axis=0)
            v_tail = ccent - Spos
        else:
            v_tail = np.array([1.0, 0.0, 0.0])
        v_anchor = project_onto_plane(v_tail, n_plane)
        if np.linalg.norm(v_anchor) < 1e-8:
            v_anchor = project_onto_plane(np.cross(n_plane, v1), n_plane)
        return Spos, unit(v_anchor)

# ——— Ammonium (cation role) ————————————————————————————————

class AmmoniumFG(FGDef):
    """
    Neutral amines → ammonium (cation role).
    Neutral pattern: [NX3;H2,H1,H0;!$([NX3](=O))]

    Anchor:
      center = N+
      vector = (N → alpha-carbon centroid), projected onto local plane when definable.
    """
    neutral_pattern = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$([NX3](=O))]")

    def __init__(self):
        super().__init__("ammonium", "cation", "[NX3;H2,H1,H0;!$([NX3](=O))]")

    def to_ionic(self, mol: Chem.Mol, match: Tuple[int, ...]) -> Chem.Mol:
        N_idx = match[0]
        N = mol.GetAtomWithIdx(N_idx)
        if N.GetFormalCharge() > 0:
            return mol
        em = Chem.EditableMol(mol)
        H_idx = em.AddAtom(Chem.Atom(1))
        mol = em.GetMol()
        conf = mol.GetConformer()
        npos = conf.GetAtomPosition(N_idx)
        conf.SetAtomPosition(H_idx, Point3D(npos.x + 1.0, npos.y, npos.z))
        mol = Chem.RWMol(mol)
        mol.AddBond(N_idx, H_idx, Chem.BondType.SINGLE)
        mol = mol.GetMol()
        mol.GetAtomWithIdx(N_idx).SetFormalCharge(+1)
        Chem.SanitizeMol(mol)
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=80)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=80)
        except Exception:
            pass
        return mol

    def to_neutral(self, mol: Chem.Mol, match: Tuple[int, ...]) -> Chem.Mol:
        # If N is +1, remove one H if available and set charge 0
        N_idx = match[0]
        N = mol.GetAtomWithIdx(N_idx)
        if N.GetFormalCharge() <= 0:
            return mol
        # find an H attached to N
        H_idx = None
        for nb in N.GetNeighbors():
            if nb.GetAtomicNum() == 1:
                H_idx = nb.GetIdx()
                break
        if H_idx is None:
            # Fallback: just zero the charge (structure may not be perfect)
            N.SetFormalCharge(0)
            Chem.SanitizeMol(mol)
            return mol
        em = Chem.EditableMol(mol)
        em.RemoveAtom(H_idx)
        mol = em.GetMol()
        mol.GetAtomWithIdx(N_idx).SetFormalCharge(0)
        Chem.SanitizeMol(mol)
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=60)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=60)
        except Exception:
            pass
        return mol

    def anchor_center_and_vector(self, mol: Chem.Mol, match: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        coords = rdconf_to_numpy(mol)
        N_idx = match[0]
        Npos  = coords[N_idx]
        neighs = [nb.GetIdx() for nb in mol.GetAtomWithIdx(N_idx).GetNeighbors()]
        if len(neighs) >= 2:
            v1 = coords[neighs[0]] - Npos; v2 = coords[neighs[1]] - Npos
            n_plane = unit(np.cross(v1, v2))
        else:
            n_plane = np.array([0.0, 0.0, 1.0])
        carbons = [i for i in neighs if mol.GetAtomWithIdx(i).GetAtomicNum() == 6]
        if carbons:
            ccent = coords[carbons].mean(axis=0)
            v_tail = ccent - Npos
        else:
            v_tail = np.array([1.0, 0.0, 0.0])
        v_proj = project_onto_plane(v_tail, n_plane)
        v_anchor = v_tail if np.linalg.norm(v_proj) < 1e-8 else v_proj
        return Npos, unit(v_anchor)

# ——— Registry & detection helpers —————————————————————————

def get_fg_registry() -> List[FGDef]:
    """Default registry (order may imply priority)."""
    return [
        CarboxylateFG(),
        PhosphonateFG(),
        SulfonateFG(),
        ThiolateFG(),
        AmmoniumFG(),
    ]

def detect_fg_matches_neutral(mol: Chem.Mol, role: str, registry: Optional[List[FGDef]] = None
                              ) -> List[Tuple[FGDef, Tuple[int, ...]]]:
    """
    Detect NEUTRAL-form matches for groups of the given role.
    This is used to index group instances; we then activate exactly one per site.
    """
    if registry is None:
        registry = get_fg_registry()
    matches: List[Tuple[FGDef, Tuple[int, ...]]] = []
    for fg in registry:
        if fg.role != role:
            continue
        # Use class-specific neutral patterns where known
        patt = None
        if isinstance(fg, CarboxylateFG):
            patt = CarboxylateFG.smarts_neutral
        elif isinstance(fg, PhosphonateFG):
            patt = PhosphonateFG.neutral_pattern
        elif isinstance(fg, SulfonateFG):
            patt = SulfonateFG.neutral_pattern
        elif isinstance(fg, ThiolateFG):
            patt = ThiolateFG.neutral_pattern
        elif isinstance(fg, AmmoniumFG):
            patt = AmmoniumFG.neutral_pattern
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt):
            matches.append((fg, match))
    return matches

def build_ionic_form(mol: Chem.Mol, role: str, ff: str = "uff",
                     registry: Optional[List[FGDef]] = None):
    """
    (Legacy utility) Convert all matching groups to ionic form.
    Kept for completeness; the main script uses per-site activation instead.
    """
    if registry is None:
        registry = get_fg_registry()
    init = detect_fg_matches_neutral(mol, role, registry)
    mwork = Chem.Mol(mol)
    seen = set()
    for fg, match in init:
        if fg.name in seen:
            continue
        mwork = fg.to_ionic(mwork, match)
        seen.add(fg.name)
    try:
        if ff == "mmff" and AllChem.MMFFHasAllMoleculeParams(mwork):
            AllChem.MMFFOptimizeMolecule(mwork, maxIters=200)
        else:
            AllChem.UFFOptimizeMolecule(mwork, maxIters=200)
    except Exception:
        pass
    # For compatibility, also return "matches" (ionic detection not used here)
    return mwork, init

