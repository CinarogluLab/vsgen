"""Protonation-state check against experimental pKa (IUPAC Digitized pKa Dataset).

Uses molecules with exactly one measured aqueous pKa at 20-25 C and exactly
one ionizable group (IONIZABLE, a tool-independent group list). Henderson-
Hasselbalch at pH 7.4 gives the expected net charge(s): both forms when the
pKa is within 1 unit of 7.4 (each >= ~9% populated), otherwise only the
dominant one; pKa values within 0.3 of a window edge accept either answer.
Reports, for vsgen's protonation step (no tautomers):
  dominant present  the expected dominant charge state was emitted
  implausible       a charge state was emitted that should be < ~9% populated
The dataset is CC BY-NC 4.0: use it for validation, do not ship it.

usage: python bench/pka_eval.py score CSV [--model molgpka|dimorphite] [--predictions tool.jsonl]
       python bench/pka_eval.py export CSV smiles.txt   (input list for bench/pka_backends.py)
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # before numpy: 16 pool workers x multithreaded BLAS thrash

import argparse
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from vsgen import chem

RDLogger.DisableLog("rdApp.*")
OPTS = None


# Groups ionizable somewhere in pH ~2-12, as center-atom SMARTS. Independent of
# both Dimorphite-DL's and MolGpKa's site lists, so neither is favored. Only
# molecules with exactly one such center are scored: there the measured pKa
# belongs to the only group that can change the net charge.
IONIZABLE = [Chem.MolFromSmarts(x) for x in (
    "[OX2H1][CX3,SX4,SX3,PX4]=O",  # carboxylic, sulfonic, sulfinic, phosphoric/phosphonic acids
    "[nH]1nnnc1", "[nH]1nncn1",  # tetrazole / triazole NH
    "[OX2H1]c", "[SX2H1]", "[OX2H1][CX3]=[CX3]", "[OX2H1]N=[CX3]", "[OX2H1][NX3]C=O",  # phenol, thiol, enol, oxime, hydroxamic
    "[NX3H1](C=O)C=O", "[NX3H1,NX3H2]S(=O)=O", "[CX4;H1,H2](C=O)C=O",  # imide, sulfonamide, 1,3-dicarbonyl CH
    "[NX3;!$(N[!#6;!#1]);!$(NC=[O,N,S]);!$(Na);!$(NC#N)][CX4]",  # aliphatic amine
    "[CX3](=[NX2;!$(N[O,N])])[NX3]", "[CX3]=[NX2;!$(N[O,N]);!$(N=CN)]",  # amidine/guanidine, imine
    "[nX2]", "[NX3;H2,H1;!$(NC=[O,S,N]);!$(NS=O)]a",  # aromatic N, aniline
)]


def ionizable_centers(m):
    return len({mt[0] for q in IONIZABLE for mt in m.GetSubstructMatches(q)})


def _base_table(path):
    d = pd.read_csv(path)
    d = d[d["T"].isin(["20", "25"]) & d["cosolvent"].isna() & d["pka_type"].isin(["pKa1", "pKaH1", "pKa2", "pKaH2"])]
    d = d[pd.to_numeric(d["pka_value"], errors="coerce").notna()].copy()
    d["pka"] = d["pka_value"].astype(float)
    return d


def _usable(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None or len(Chem.GetMolFrags(m)) > 1 or any(a.GetFormalCharge() for a in m.GetAtoms()):
        return None  # neutral single-component inputs only, so the expected charge is unambiguous
    if not 3 <= m.GetNumHeavyAtoms() <= 60 or not set(a.GetSymbol() for a in m.GetAtoms()) <= set("C N O S P F Cl Br I".split()):
        return None
    return m


def load(path):
    """Single-group set: (smiles, pKa, charge of the ionized form)."""
    d = _base_table(path)
    n_types = d.groupby("InChI")["pka_type"].nunique()
    d = d[d["InChI"].isin(n_types[n_types == 1].index) & d["pka_type"].isin(["pKa1", "pKaH1"])]
    d = d.groupby(["InChI", "pka_type"]).agg(smiles=("SMILES", "first"), pka=("pka", "median")).reset_index()
    rows = []
    for _, r in d.iterrows():
        m = _usable(r["smiles"])
        if m is None or ionizable_centers(m) != 1:
            continue
        rows.append((Chem.MolToSmiles(m), r["pka"], -1 if r["pka_type"] == "pKa1" else +1))
    return rows


def load_two_site(path):
    """Diacids (pKa1, pKa2) and dibases (pKaH1, pKaH2) with exactly two
    ionizable groups: (smiles, (pKa_low, pKa_high), sign)."""
    d = _base_table(path)
    types = d.groupby("InChI")["pka_type"].agg(lambda t: frozenset(t))
    rows = []
    for inchi, t in types.items():
        if t not in (frozenset({"pKa1", "pKa2"}), frozenset({"pKaH1", "pKaH2"})):
            continue
        g = d[d["InChI"] == inchi]
        m = _usable(g["SMILES"].iloc[0])
        if m is None or ionizable_centers(m) != 2:
            continue
        vals = sorted(g.groupby("pka_type")["pka"].median())
        rows.append((Chem.MolToSmiles(m), tuple(vals), -1 if "pKa1" in t else +1))
    return rows


def expected(pka, charged):
    """Set of acceptable net charges at pH 7.4, and the dominant one."""
    frac_charged = 1 / (1 + 10 ** ((pka - 7.4) if charged < 0 else (7.4 - pka)))
    dominant = charged if frac_charged >= 0.5 else 0
    near_edge = abs(abs(pka - 7.4) - 1.0) <= 0.3
    ok = {charged, 0} if abs(pka - 7.4) <= 1.0 or near_edge else {dominant}
    return dominant, ok


def expected_two(pkas, sign, ph=7.4, min_pop=0.05):
    """Macro populations of charges 0, sign, 2*sign from two macro pKa values."""
    lo, hi = pkas
    if sign < 0:  # H2A -> HA- (lo) -> A2- (hi)
        logw = {0: 0.0, -1: ph - lo, -2: 2 * ph - lo - hi}
    else:  # B -> BH+ (hi) -> BH2 2+ (lo)
        logw = {0: 0.0, 1: hi - ph, 2: hi + lo - 2 * ph}
    w = {q: 10 ** v for q, v in logw.items()}
    tot = sum(w.values())
    pop = {q: v / tot for q, v in w.items()}
    return max(pop, key=pop.get), {q for q, p in pop.items() if p >= min_pop}


def charges_from_prediction(smi, pred, o):
    """Net charges emitted for one molecule from an external tool's record."""
    if not pred or pred.get("error"):
        return None
    if pred["kind"] == "states":
        return {Chem.GetFormalCharge(Chem.MolFromSmiles(s)) for s, p in pred["states"] if p >= pred.get("min_pop", 0.03)}
    mol = Chem.MolFromSmiles(smi)
    acids = {int(k): v for k, v in pred["acids"].items()}
    bases = {int(k): v for k, v in pred["bases"].items()}
    return {Chem.GetFormalCharge(m) for m in chem.states_from_sites(mol, acids, bases, o, couple=not pred.get("coupled"))}


def macro_pka(pred, charged):
    """First macro pKa of the measured type, or None if the tool predicts no such site."""
    if not pred or pred.get("error"):
        return None
    if pred["kind"] == "states" and pred.get("macro"):
        return pred["macro"].get("acid" if charged < 0 else "base")
    sites = pred.get("acids", {}) if charged < 0 else pred.get("bases", {})
    if not sites:
        return None
    return min(sites.values()) if charged < 0 else max(sites.values())


def _conn_key(smi):
    from rdkit.Chem.MolStandardize import rdMolStandardize

    m = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.FragmentParent(Chem.MolFromSmiles(smi)))
    return Chem.MolToInchiKey(m).split("-")[0]


def emitted(smi):
    mols = chem.protomers(smi, OPTS)
    return {Chem.GetFormalCharge(m) for m in mols}


def molgpka_record(smi):
    from vsgen import pka
    from rdkit.Chem.MolStandardize import rdMolStandardize

    mol = rdMolStandardize.Uncharger().uncharge(Chem.MolFromSmiles(smi))
    _, a, b = pka.predict(mol)
    return {"kind": "sites", "acids": a, "bases": b}


def init(o):
    global OPTS
    OPTS = o


def _report(name, rows, states, macros, two_rows, two_states):
    dom_ok, impl, n_states, by_group, err = [], [], [], {}, []
    failed = sum(s is None for s in states)
    for (smi, pka, charged), got, mp in zip(rows, states, macros):
        if got is None:
            got = set()
        dominant, ok = expected(pka, charged)
        dom_ok.append(dominant in got)
        impl.append(bool(got - ok))
        n_states.append(len(got))
        if mp is not None:
            err.append(abs(mp - pka))
        g = "acid" if charged < 0 else "base"
        band = "within 1 of 7.4" if abs(pka - 7.4) <= 1 else ("pKa < 6.4" if pka < 6.4 else "pKa > 8.4")
        by_group.setdefault((g, band), []).append((dominant in got, bool(got - ok)))
        size = "heavy atoms >= 12" if Chem.MolFromSmiles(smi).GetNumHeavyAtoms() >= 12 else "heavy atoms < 12"
        by_group.setdefault(("all", size), []).append((dominant in got, bool(got - ok)))
    print(f"== {name}: {len(rows)} single-group molecules ({failed} without a prediction, scored as no states)")
    print(f"dominant state emitted: {np.mean(dom_ok):.1%}   implausible state emitted: {np.mean(impl):.1%}   "
          f"charge states per molecule: {np.mean(n_states):.2f}")
    if err:
        e = np.array(err)
        print(f"pKa error vs experiment: MAE {e.mean():.2f}, RMSE {np.sqrt((e ** 2).mean()):.2f}, "
              f"|error| <= 1: {np.mean(e <= 1):.1%}  (site predicted for {len(err)}/{len(rows)})")
    for k in sorted(by_group):
        v = np.array(by_group[k])
        print(f"  {k[0]:4s} {k[1]:16s} n={len(v):5d}  dominant {v[:, 0].mean():6.1%}  implausible {v[:, 1].mean():6.1%}")
    if two_rows:
        d2, i2 = [], []
        for (smi, pkas, sign), got in zip(two_rows, two_states):
            got = got or set()
            dominant, ok = expected_two(pkas, sign)
            d2.append(dominant in got)
            i2.append(bool(got - ok))
        print(f"two-site set ({len(two_rows)} diacids/dibases): dominant {np.mean(d2):.1%}, implausible {np.mean(i2):.1%}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="write the benchmark SMILES for external tools")
    e.add_argument("csv")
    e.add_argument("out")
    s = sub.add_parser("score")
    s.add_argument("csv")
    s.add_argument("--model", default="molgpka", choices=["molgpka", "dimorphite"])
    s.add_argument("--predictions", help="JSONL from bench/pka_backends.py (smiles -> record)")
    s.add_argument("--precision", type=float, default=1.0, help="Dimorphite-DL only")
    s.add_argument("--margin", type=float, default=0.5, help="window widening for site-based predictors")
    s.add_argument("--exclude-keys", help="InChIKey connectivity blocks to drop (e.g. tools' training sets)")
    a = ap.parse_args()
    rows, two_rows = load(a.csv), load_two_site(a.csv)
    if a.cmd == "score" and a.exclude_keys:
        drop = set(open(a.exclude_keys).read().split())
        rows = [r for r in rows if _conn_key(r[0]) not in drop]
        two_rows = [r for r in two_rows if _conn_key(r[0]) not in drop]
    if a.cmd == "export":
        with open(a.out, "w") as f:
            for smi in dict.fromkeys([r[0] for r in rows] + [r[0] for r in two_rows]):
                f.write(smi + "\n")
        print(f"wrote {len(rows)} single-group + {len(two_rows)} two-site molecules to {a.out}")
        return
    o = chem.Options(pka_precision=a.precision, pka_model=a.model, pka_margin=a.margin)
    smiles = [r[0] for r in rows] + [r[0] for r in two_rows]
    if a.predictions:
        preds = {}
        for line in open(a.predictions):
            rec = json.loads(line)
            preds[rec["smiles"]] = rec
        states = [charges_from_prediction(s, preds.get(s), o) for s in smiles]
        macros = [macro_pka(preds.get(r[0]), r[2]) for r in rows]
        name = os.path.basename(a.predictions)
    else:
        with Pool(16, initializer=init, initargs=(o,)) as p:
            states = p.map(emitted, smiles, chunksize=50)
            if a.model == "molgpka":
                recs = p.map(molgpka_record, [r[0] for r in rows], chunksize=50)
            else:  # Dimorphite-DL's own table, so its pKa can be scored too (see pka_external)
                from pka_external import dimorphite_records

                table = dimorphite_records([r[0] for r in rows])
                recs = [table[r[0]] for r in rows]
        macros = [macro_pka(rec, r[2]) for rec, r in zip(recs, rows)]
        if a.model == "dimorphite":
            from pka_external import dimorphite_version

            name = f"{dimorphite_version()} precision {a.precision}"
        else:
            name = f"molgpka margin {a.margin}"
    _report(name, rows, states[: len(rows)], macros, two_rows, states[len(rows):])


if __name__ == "__main__":
    main()
