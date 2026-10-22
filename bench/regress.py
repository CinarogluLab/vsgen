"""UFF-protocol regression: fast mode vs thorough (full-minimization) mode.

This only checks that fast mode finds UFF minima as low as thorough mode does.
It is NOT evidence of docking quality; bench/dockval.py tests that.
"Not worse" = fast energy at most 1 kcal/mol above thorough (one-sided: a
lower energy is not a defect). Variants of one parent are correlated, so the
95% CI is a bootstrap over parents (_Name).
Pass: success loss <= 0.5 pp and CI lower bound of "not worse" >= 92.4%.

usage: python bench/regress.py [bench/variants500.sdf] [--procs 16]
"""

import argparse
import time
from multiprocessing import get_context

import numpy as np
from rdkit import Chem

from vsgen.chem import variant_seed
from vsgen.conf import Conf3D, build_cpu



def job(args):
    smi, seed, cfg = args
    models, _ = build_cpu(smi, seed, cfg)
    return models[0][1] if models else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="?", default="bench/variants500.sdf")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--force-field", default="uff", choices=["uff", "mmff94s"])
    a = ap.parse_args()
    ff = a.force_field
    CONFIGS = {
        "thorough_ref": (Conf3D(thorough=True, ring_confs=1, force_field=ff), 0),
        "thorough_seed2": (Conf3D(thorough=True, ring_confs=1, force_field=ff), 1),  # noise floor
        "fast": (Conf3D(ring_confs=1, force_field=ff), 0),
    }
    recs = [(m.GetProp("SMILES"), m.GetProp("_Name")) for m in Chem.SDMolSupplier(a.variants) if m is not None]
    smiles, parent = [s for s, _ in recs], [n for _, n in recs]
    order = sorted(range(len(smiles)), key=lambda i: -Chem.AddHs(Chem.MolFromSmiles(smiles[i])).GetNumAtoms())
    res, wall = {}, {}
    with get_context("spawn").Pool(a.procs) as pool:
        for name, (cfg, salt) in CONFIGS.items():
            t = time.perf_counter()
            out = pool.map(job, [(smiles[i], variant_seed(42 + salt, smiles[i]), cfg) for i in order], chunksize=1)
            wall[name] = time.perf_counter() - t
            res[name] = dict(zip(order, out))
    ref = res["thorough_ref"]
    ok_ref = sum(e is not None for e in ref.values())
    print(f"{'config':16s} {'wall':>6s} {'speedup':>8s} {'success':>8s} {'not worse':>10s} {'95% CI':>13s} "
          f"{'|dE|<=1':>8s} {'p95 |dE|':>8s}")
    rng = np.random.default_rng(0)
    passed = True
    for name, r in res.items():
        both = [i for i in r if r[i] is not None and ref[i] is not None]
        d = np.array([r[i] - ref[i] for i in both])
        ok = sum(e is not None for e in r.values())
        groups = {}
        for i, x in zip(both, d <= 1):
            groups.setdefault(parent[i], []).append(x)
        g = list(groups.values())
        boots = [np.mean(np.concatenate([g[k] for k in rng.integers(len(g), size=len(g))])) for _ in range(1000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"{name:16s} {wall[name]:6.1f} {wall['thorough_ref'] / wall[name]:7.2f}x {ok:8d} "
              f"{np.mean(d <= 1):10.1%} {lo:6.1%}-{hi:6.1%} {np.mean(np.abs(d) <= 1):8.1%} "
              f"{np.percentile(np.abs(d), 95):8.2f}")
        if name == "fast":
            passed = lo >= 0.924 and (ok_ref - ok) / len(smiles) <= 0.005
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
