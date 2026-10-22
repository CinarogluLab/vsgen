"""Intramolecular ion pairs in vsgen's 3D models of zwitterionic variants.

The MMFF94s dielectric (eps = 4r) is there so a cation and an anion in one
molecule do not fold onto each other as they would in vacuum. This builds the
kept model of every vsgen variant that carries both a cation and an anion (not
bonded to each other, so nitro groups and N-oxides do not count) and reports
the closest cation-anion heavy-atom distance, split by how many bonds separate
the pair: close pairs (<= 4 bonds) cannot avoid contact, distant ones can.
A contact under 3.0 A is a salt bridge or a charge-assisted H-bond, not
necessarily an artifact, so distributions are reported, not a pass mark.
Run it once per vsgen version to compare (PYTHONPATH selects the version).

usage: python bench/ionpair.py corpus.smi [--procs 16] [--out pairs.jsonl]
"""

import argparse
import json
from multiprocessing import Pool

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
BINS = [(0, 4, "<= 4 bonds"), (5, 7, "5-7 bonds"), (8, 999, ">= 8 bonds")]


def ions(mol):
    def isolated(a, sign):  # charged, and not bonded to an opposite charge (nitro, N-oxide, ylide)
        return a.GetFormalCharge() * sign > 0 and not any(n.GetFormalCharge() * sign < 0 for n in a.GetNeighbors())
    cat = [a.GetIdx() for a in mol.GetAtoms() if isolated(a, +1)]
    an = [a.GetIdx() for a in mol.GetAtoms() if isolated(a, -1)]
    return cat, an


def zwitterions(smi_name):
    from vsgen.chem import prepare_2d

    smi, name = smi_name
    variants, _ = prepare_2d(0, name, smi)
    out = []
    for v in variants:
        cat, an = ions(Chem.MolFromSmiles(v.smiles))
        if cat and an:
            out.append((v.smiles, v.seed, name))
    return out


def measure(job):
    from vsgen.conf import build_cpu

    smi, seed, name = job
    models, _ = build_cpu(smi, seed)
    if not models:
        return None
    mol = models[0][0]  # the kept ("best") model; H were added after the heavy atoms, so indices match
    pos = mol.GetConformer().GetPositions()
    topo = Chem.GetDistanceMatrix(mol)
    cat, an = ions(Chem.MolFromSmiles(smi))
    d, i, j = min((float(np.linalg.norm(pos[i] - pos[j])), i, j) for i in cat for j in an)
    return {"name": name, "smiles": smi, "closest_A": round(d, 3), "bonds": int(topo[i][j]),
            "force_field": mol.GetProp("force_field")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("smi")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = [tuple((l.split() + [""])[:2]) for l in open(a.smi) if l.strip()]
    with Pool(a.procs) as p:
        jobs = [z for zs in p.map(zwitterions, rows, chunksize=20) for z in zs]
        res = [r for r in p.map(measure, jobs, chunksize=2) if r]
    import vsgen

    mmff = [r for r in res if r["force_field"] == "MMFF94s"]  # only these test the MMFF dielectric
    print(f"{len(rows)} inputs, {len(jobs)} zwitterionic variants, {len(res)} built, {len(mmff)} with MMFF94s "
          f"(the table; {len(res) - len(mmff)} UFF/none left out); vsgen from {vsgen.__file__}")
    print(f"{'separation':12s} {'n':>5s} {'< 3.0 A':>8s} {'< 4.0 A':>8s} {'median':>7s} {'25th pct':>8s}")
    for lo, hi, label in BINS:
        d = np.array([r["closest_A"] for r in mmff if lo <= r["bonds"] <= hi])
        if len(d):
            print(f"{label:12s} {len(d):5d} {np.mean(d < 3):8.1%} {np.mean(d < 4):8.1%} {np.median(d):7.2f} "
                  f"{np.percentile(d, 25):8.2f}")
    if a.out:
        with open(a.out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in res)


if __name__ == "__main__":
    main()
