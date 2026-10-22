"""Per-atom pKa prediction with MolGpKa, ported to NumPy (no PyTorch needed).

MolGpKa (Pan et al., J. Chem. Inf. Model. 2021; MIT, https://github.com/Xundrug/MolGpKa)
is a graph-convolutional network with separate acid and base models. Site
detection and atom features are taken from its code unchanged; the forward
pass reimplements its GCNNet (5 GCN layers + batch norm, attention pooling,
3 dense layers). bench/convert_molgpka.py made the weight archives, and
tests/test_pka.py checks this port against the PyTorch original's outputs.
"""

import csv
import os
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

DATA = os.path.join(os.path.dirname(__file__), "data")
_HYB = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2]
_ELEMENTS = ["C", "H", "O", "N", "S", "Cl", "F", "Br", "P", "I"]


@lru_cache(maxsize=None)
def _patterns():
    acid, base = [], []
    with open(os.path.join(DATA, "molgpka_smarts.tsv")) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            row = {k.strip(): v.strip() for k, v in row.items()}
            idx = [int(i) for i in row["Index"].split(",")]
            (acid if row["Acid_or_base"] == "A" else base).append((Chem.MolFromSmarts(row["SMARTS"]), idx))
    return acid, base


@lru_cache(maxsize=None)
def _weights(kind):
    return dict(np.load(os.path.join(DATA, f"molgpka_{kind}.npz")))


def _unique(matches):
    singles = sorted({m[0] for m in matches if len(m) == 1})
    out = [m for m in matches if len(m) == 2] + [[s] for s in singles]
    return [a for m in out for a in m]


def sites(mol_h):
    """Ionizable atoms of an H-added molecule, as MolGpKa defines them:
    acid sites are the acidic H atoms, base sites the atoms that accept H+."""
    acid_p, base_p = _patterns()
    acid, base = [], []
    for pat, idx in acid_p:
        for m in mol_h.GetSubstructMatches(pat):
            acid.append([m[i] for i in idx] if len(idx) == 2 else [m[idx[0]]])
    for pat, idx in base_p:
        for m in mol_h.GetSubstructMatches(pat):
            base.extend([m[i]] for i in idx)
    return _unique(acid), _unique(base)


def _features(mol_h, aid):
    AllChem.ComputeGasteigerCharges(mol_h)
    Chem.AssignStereochemistry(mol_h)
    ring = mol_h.GetRingInfo()
    dist = Chem.GetDistanceMatrix(mol_h)[aid]
    rows = []
    for i, atom in enumerate(mol_h.GetAtoms()):
        sym = atom.GetSymbol() if atom.GetSymbol() in _ELEMENTS else _ELEMENTS[-1]
        hyb = atom.GetHybridization() if atom.GetHybridization() in _HYB else _HYB[-1]
        rows.append([s == sym for s in _ELEMENTS] + [atom.GetDegree()] + [h == hyb for h in _HYB]
                    + [atom.GetValence(Chem.ValenceType.IMPLICIT), atom.GetIsAromatic()]
                    + [ring.IsAtomInRingOfSize(i, n) for n in range(3, 9)]
                    # MolGpKa's donor/acceptor flags test an int against a list of match
                    # tuples, so they were always False in training; reproduce that.
                    + [False, False, atom.GetFormalCharge()]
                    # MolGpKa used len(GetShortestPath) = bond distance + 1 for other atoms, 0 for the site
                    + [0 if i == aid else dist[i] + 1, i == aid])
    return np.asarray(rows, dtype=np.float32)


def _norm_adj(mol_h):
    n = mol_h.GetNumAtoms()
    a = Chem.GetAdjacencyMatrix(mol_h).astype(np.float32) + np.eye(n, dtype=np.float32)
    d = 1.0 / np.sqrt(a.sum(1))
    return a * d[:, None] * d[None, :]


def _forward(w, x, adj):
    for k in range(1, 6):
        x = np.maximum(adj @ (x @ w[f"conv{k}.weight"]) + w[f"conv{k}.bias"], 0)
        x = (x - w[f"bn{k}.running_mean"]) / np.sqrt(w[f"bn{k}.running_var"] + 1e-5) * w[f"bn{k}.weight"] + w[f"bn{k}.bias"]
    gate = x @ w["att.gate_nn.weight"].T + w["att.gate_nn.bias"]
    gate = np.exp(gate - gate.max())
    x = (gate / gate.sum() * x).sum(0)
    x = np.maximum(x @ w["fc2.weight"].T + w["fc2.bias"], 0)
    x = np.maximum(x @ w["fc3.weight"].T + w["fc3.bias"], 0)
    return float((x @ w["fc4.weight"].T + w["fc4.bias"])[0])


def predict(mol):
    """pKa of each ionizable site of a neutral molecule.

    Returns (mol_h, acids, bases): the H-added molecule, and dicts mapping heavy
    atom index -> pKa. An acid entry is the atom that loses H+ (MolGpKa
    predicts per acidic H; equivalent H on one atom keep the lowest value).
    """
    mol_h = Chem.AddHs(mol)
    acid_h, base_idx = sites(mol_h)
    adj = _norm_adj(mol_h)
    acids, bases = {}, {}
    for h in acid_h:
        heavy = mol_h.GetAtomWithIdx(h).GetNeighbors()[0].GetIdx()
        p = _forward(_weights("acid"), _features(mol_h, h), adj)
        acids[heavy] = min(p, acids.get(heavy, p))
    for b in base_idx:
        bases[b] = _forward(_weights("base"), _features(mol_h, b), adj)
    return mol_h, acids, bases
