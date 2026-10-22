"""Run external pKa predictors on a SMILES list and write JSONL for bench/pka_eval.py.

Records (one per input SMILES, keyed by it):
  {"smiles", "kind": "sites", "acids": {heavy atom: pKa}, "bases": {...}, "coupled": bool}
  {"smiles", "kind": "states", "states": [[smiles, population]], "macro": {"acid": pKa, "base": pKa}}
  {"smiles", "error": "..."}
Atom indices refer to the input SMILES's heavy atoms. "coupled" means the tool's
pKa already include the effect of neighboring charges (sequential predictors).

Run in an environment with the tools installed (not vsgen's): torch 2.6,
torch_geometric 2.8, pkasolver checkout (PKASOLVER_DIR), qupkake, unipkainfer,
molscrub (MOLSCRUB_DIR for its data files, scrub.py on PATH),
Uni-pKa checkout (UNIPKA_DIR, for its microstate enumerator) and its weights
(UNIPKA_WEIGHTS: dir with fold_1/checkpoint_best.pt, or UNIPKA_CHECKPOINT: one file
in unipkainfer's key layout; the official t_dwar_v_novartis_a_b.pt needs its keys
prefixed "unimol." and head "chembl_all" renamed "chembl"). The runtime patches below
make 2022-23 code run on current PyTorch/PyG/NumPy; each says what it fixes.

usage: python bench/pka_backends.py {pkasolver,qupkake,unipka} smiles.txt out.jsonl [--procs N]
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def _skeleton(mol):
    rw = Chem.RWMol(Chem.RemoveAllHs(mol, sanitize=False))
    for a in rw.GetAtoms():
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
    for b in rw.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    return m


def _heavy_map(tool_mol, input_mol):
    """tool heavy-atom index -> input heavy-atom index (same graph, any protonation)."""
    match = _skeleton(input_mol).GetSubstructMatch(_skeleton(tool_mol))
    return dict(enumerate(match)) if len(match) == input_mol.GetNumAtoms() else None


def _heavy_rank(mol_h, idx):
    """Index among heavy atoms of atom idx in a molecule that may contain H."""
    return sum(1 for a in mol_h.GetAtoms() if a.GetIdx() < idx and a.GetAtomicNum() > 1)


# ---------------- pkasolver ----------------

_PKS = None


def _pkasolver_init():
    """pkasolver-lite (public model). Patches: PyG 2.0.1 GIN layout for the
    saved weights (legacy GIN), and a NumPy-2-safe ensemble average."""
    global _PKS
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # pkasolver mixes devices when a GPU is visible; its 25 small models run fine on CPU
    # pkasolver passes Dimorphite-DL results through ./test.pkl: one directory per worker, or
    # parallel workers silently read each other's molecules
    os.chdir(tempfile.mkdtemp(prefix="pkasolver_", dir=os.environ.get("PKA_TMP")))
    import torch
    import torch.nn.functional as F
    import torch_geometric.nn.models as models
    from torch.nn import BatchNorm1d, Linear, ModuleList, ReLU, Sequential
    from torch_geometric.nn import GINConv

    class LegacyGIN(torch.nn.Module):  # torch_geometric 2.0.1's GIN, which the weights were saved from
        def __init__(self, in_channels, hidden_channels, num_layers, out_channels=None, dropout=0.0, **_):
            super().__init__()
            self.dropout = dropout
            self.convs = ModuleList(
                GINConv(Sequential(Linear(in_channels if i == 0 else hidden_channels, hidden_channels),
                                   BatchNorm1d(hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels)))
                for i in range(num_layers))
            self.lin = Linear(hidden_channels, out_channels)

        def forward(self, x, edge_index, *a, **k):
            for conv in self.convs:
                x = F.dropout(F.relu(conv(x, edge_index)), p=self.dropout, training=self.training)
            return self.lin(x)

    models.GIN = LegacyGIN
    torch.set_num_threads(2)
    sys.path.insert(0, os.environ["PKASOLVER_DIR"])
    # pkasolver runs its bundled Dimorphite-DL as `python script.py`: make that this interpreter
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [os.environ["PKASOLVER_DIR"], os.environ.get("PYTHONPATH")]))
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
    from pkasolver import query

    def predict_pka_value(self, loader):  # original used float() on 1-element arrays (NumPy 2 error)
        (data,) = list(loader)
        preds = torch.stack([m(x_p=data.x_p, x_d=data.x_d, edge_attr_p=data.edge_attr_p,
                               edge_attr_d=data.edge_attr_d, data=data).reshape(-1).detach() for m in self.models])
        return [float(preds.mean(0)[0]), float(preds.std(0, unbiased=False)[0])]

    query.QueryModel.predict_pka_value = predict_pka_value
    _PKS = query


def _pkasolver_one(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        states = _PKS.calculate_microstate_pka_values(mol, only_dimorphite=False)
        acids, bases = {}, {}
        for st in states:
            prot = st.protonated_mol
            hmap = _heavy_map(prot, mol)
            if hmap is None:
                return {"smiles": smi, "error": "atom mapping failed"}
            center = prot.GetAtomWithIdx(st.reaction_center_idx)
            site = hmap[_heavy_rank(prot, st.reaction_center_idx)]
            if center.GetFormalCharge() > 0:
                bases[site] = max(bases.get(site, -99), st.pka)  # conjugate acid of a base
            else:
                acids[site] = min(acids.get(site, 99), st.pka)
        return {"smiles": smi, "kind": "sites", "acids": acids, "bases": bases, "coupled": True}
    except Exception as exc:
        return {"smiles": smi, "error": f"{type(exc).__name__}: {exc}"[:200]}


# ---------------- QupKake ----------------

def _qupkake_patch_source():
    """Python run before QupKake's CLI. Patches: PyG >= 2.5 transforms need
    forward(); PyTorch 2.6 weights_only loading of QupKake's own dataset cache."""
    return r'''
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import multiprocessing
multiprocessing.cpu_count = lambda: 1  # QupKake runs xtb with -P cpu_count(); chunks already run side by side
import sys, torch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import EdgeStorage, GlobalStorage, NodeStorage
from rdkit.Chem.rdchem import Mol
from qupkake import mol_dataset, transforms
torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage, NodeStorage, EdgeStorage, Mol]
    + [c for c in vars(mol_dataset).values() if isinstance(c, type) and c.__module__ == mol_dataset.__name__])
for c in vars(transforms).values():
    if isinstance(c, type) and c.__module__ == transforms.__name__ and "__call__" in vars(c):
        c.forward = lambda self, data, _c=c: _c.__call__(self, data)
        c.__abstractmethods__ = frozenset()
from qupkake.cli import main
sys.exit(main())
'''


def _qupkake_chunk(args):
    chunk_id, items, work = args
    root = os.path.join(work, f"chunk{chunk_id}")
    os.makedirs(root, exist_ok=True)
    csv = os.path.join(root, "in.csv")
    with open(csv, "w") as f:
        f.write("name,smiles\n")
        for i, smi in items:
            f.write(f"m{i},{smi}\n")
    runner = os.path.join(root, "run.py")
    open(runner, "w").write(_qupkake_patch_source())
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    subprocess.run([sys.executable, runner, "file", csv, "-r", root, "-o", "out.sdf"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=root, timeout=7200, env=env)
    sites = {i: ({}, {}) for i, _ in items}
    out = os.path.join(root, "output", "out.sdf")
    ok_names = set()
    if os.path.exists(out):
        for m in Chem.SDMolSupplier(out, removeHs=False, sanitize=False):
            if m is None:
                continue
            i = int(m.GetProp("_Name")[1:])
            ok_names.add(i)
            idx = int(m.GetProp("idx"))
            inp = Chem.MolFromSmiles(dict(items)[i])
            hmap = _heavy_map(m, inp)
            if hmap is None:
                continue
            site = hmap[_heavy_rank(m, idx)]
            pka = float(m.GetProp("pka"))
            acids, bases = sites[i]
            if m.GetProp("pka_type").startswith("acid"):
                acids[site] = min(acids.get(site, 99), pka)
            else:
                bases[site] = max(bases.get(site, -99), pka)
    processed = {f[:-3] for f in os.listdir(os.path.join(root, "processed"))} if os.path.isdir(os.path.join(root, "processed")) else set()
    recs = []
    for i, smi in items:
        if i not in ok_names and f"m{i}" not in processed:
            recs.append({"smiles": smi, "error": "QupKake did not process the molecule"})
        else:
            acids, bases = sites[i]
            recs.append({"smiles": smi, "kind": "sites", "acids": acids, "bases": bases, "coupled": False})
    return recs


# ---------------- molscrub ----------------

def _molscrub_reaction_kinds(pka_file):
    """{reaction name: "acid"|"base"} from molscrub's pka_reactions.txt.

    Lines are `SMARTS_acid <<>> SMARTS_base name pKa`; the acid side is the
    protonated form, so a cationic acid side means the pKa belongs to a base
    (BH+ -> B) and anything else is an acid (HA -> A-)."""
    kinds = {}
    for line in open(pka_file):
        if line.startswith("#") or "<<>>" not in line:
            continue
        acid_smarts, rest = line.split("<<>>")
        parts = rest.split()
        name = parts[-2]
        query = Chem.MolFromSmarts(acid_smarts.strip())
        cationic = query is not None and sum(a.GetFormalCharge() for a in query.GetAtoms()) > 0
        kinds[name] = "base" if cationic else "acid"
    return kinds


def run_molscrub(smiles, out, procs=8, ph=7.4, ph_low=None, ph_high=None):
    """molscrub writes 3D states to SDF, with the pKa of each site it considered
    in `pka_props_*`. Record both: the states (for charge-state scoring) and the
    per-site pKa (for the pKa error)."""
    import ast

    molscrub_dir = os.environ["MOLSCRUB_DIR"]
    kinds = _molscrub_reaction_kinds(os.path.join(molscrub_dir, "molscrub", "data", "pka_reactions.txt"))
    work = tempfile.mkdtemp(prefix="molscrub_", dir=os.path.dirname(os.path.abspath(out)))
    try:
        src, sdf = os.path.join(work, "in.smi"), os.path.join(work, "out.sdf")
        with open(src, "w") as f:
            for i, smi in enumerate(smiles):
                f.write(f"{smi} m{i}\n")
        window = ["--ph_low", str(ph_low), "--ph_high", str(ph_high)] if ph_low is not None else []
        subprocess.run([os.path.join(os.path.dirname(sys.executable), "scrub.py"), src, "-o", sdf,
                        "--cpu", str(procs), "--ph", str(ph), *window], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=14400)
        states, sites = {}, {}
        for mol in Chem.SDMolSupplier(sdf, removeHs=False):
            if mol is None:
                continue
            idx = int(mol.GetProp("_Name")[1:].split("_")[0])  # "m12_i0": input 12, isomer 0
            inp = Chem.MolFromSmiles(smiles[idx])
            state = Chem.RemoveHs(mol)
            states.setdefault(idx, []).append(Chem.MolToSmiles(state))
            if idx in sites:
                continue  # the pKa list is a property of the input, not of each state
            hmap = _heavy_map(state, inp)
            acids, bases = {}, {}
            for key, value in mol.GetPropsAsDict().items():
                if not key.startswith("pka_props"):
                    continue
                prop = ast.literal_eval(value)
                site = hmap.get(_heavy_rank(mol, prop["atom"])) if hmap else None
                if site is None:
                    continue
                (bases if kinds.get(prop["name"]) == "base" else acids)[site] = float(prop["pKa"])
            sites[idx] = (acids, bases)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    with open(out, "w") as f:
        for i, smi in enumerate(smiles):
            if i not in states:
                f.write(json.dumps({"smiles": smi, "error": "molscrub produced no state"}) + "\n")
                continue
            acids, bases = sites.get(i, ({}, {}))
            pop = 1.0 / len(states[i])
            f.write(json.dumps({"smiles": smi, "kind": "states", "states": [[s, pop] for s in states[i]],
                                "acids": acids, "bases": bases, "coupled": False}) + "\n")


# ---------------- Uni-pKa ----------------

MU = 6.504894871171601  # Uni-pKa's pKa target normalization (unimol_pka.py, dwar_8228 mean)


def _unipka_setup():
    """Patches: unipkainfer wrote LMDB keys unpadded while reading them in
    lexicographic cursor order, misassigning results in batches > 10; and its
    checkpoint loader used weights_only=False (the pinned checkpoint only holds
    tensors, NumPy scalars/dtypes, argparse.Namespace and OrderedDict)."""
    import _codecs
    import argparse as _argparse
    import pickle

    import torch
    from unipkainfer.pka_predictor import free_energy as fe
    from unipkainfer.unicoreinfer import checkpoint_utils

    dtypes = [getattr(np.dtypes, n) for n in dir(np.dtypes) if n.endswith("DType")]
    torch.serialization.add_safe_globals([_argparse.Namespace, _codecs.encode, np.dtype,
                                          np._core.multiarray.scalar, *dtypes])

    class _SafeTorch:
        def __getattr__(self, name):
            return getattr(torch, name)

        @staticmethod
        def load(*a, **k):
            k["weights_only"] = True
            return torch.load(*a, **k)

    checkpoint_utils.torch = _SafeTorch()

    def write_lmdb(smiles, task_name, processed_dir, cfg):
        import lmdb

        path = processed_dir / task_name / f"{fe._UNIPKA_VALID_SUBSET}.lmdb"
        if path.exists():
            path.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)
        env = lmdb.open(str(path), subdir=False, readonly=False, lock=False, readahead=False,
                        meminit=False, max_readers=1, map_size=int(100e9))
        try:
            with env.begin(write=True) as txn:
                for idx, smi in enumerate(smiles):
                    txn.put(f"{idx:012d}".encode("ascii"),
                            pickle.dumps(fe._smiles_to_free_energy_record(smi, cfg), protocol=-1))
        finally:
            env.close()

    fe._write_free_energy_lmdb = write_lmdb
    if os.environ.get("UNIPKA_CHECKPOINT"):  # a local checkpoint in unipkainfer's layout (e.g. the converted official one)
        from pathlib import Path

        fe.ensure_checkpoint = lambda model_dir, fold, **k: Path(os.environ["UNIPKA_CHECKPOINT"])
    sys.path.insert(0, os.path.join(os.environ["UNIPKA_DIR"], "enumerator"))
    import main as enumerator

    # simple_smarts_pattern.tsv (Uni-pKa's ensemble default) lacks imides, rhodanines and
    # N-heteroaryl amides; smarts_pattern.tsv covers them (UNIPKA_TEMPLATE=smarts_pattern.tsv)
    name = os.environ.get("UNIPKA_TEMPLATE", "simple_smarts_pattern.tsv")
    template = enumerator.read_template(os.path.join(os.environ["UNIPKA_DIR"], "enumerator", name))
    return enumerator, template


def _ensemble(enumerator, template, smi, span=2):
    q0 = Chem.GetFormalCharge(Chem.MolFromSmiles(smi))
    ens, s0 = {}, [smi]
    s0, b = enumerator.enumerate_template(s0, *template, maxiter=10, mode="A")
    if b:
        ens[q0 - 1] = b
        for q in range(q0 - 2, q0 - span - 1, -1):
            if q + 1 in ens:
                _, bb = enumerator.enumerate_template(ens[q + 1], *template, maxiter=10, mode="A")
                if bb:
                    ens[q] = bb
    a, s0 = enumerator.enumerate_template(s0, *template, maxiter=10, mode="B")
    if a:
        ens[q0 + 1] = a
        for q in range(q0 + 2, q0 + span + 1):
            if q - 1 in ens:
                aa, _ = enumerator.enumerate_template(ens[q - 1], *template, maxiter=10, mode="B")
                if aa:
                    ens[q] = aa
    ens[q0] = s0
    return q0, ens


def run_unipka(smiles, out, chunk=40, ph=7.4):
    from unipkainfer import UnipkaFreeEnergyConfig, predict_standard_free_energies

    enumerator, template = _unipka_setup()
    cfg = UnipkaFreeEnergyConfig(model_dir=os.environ.get("UNIPKA_WEIGHTS", "."), nthreads=16)
    with open(out, "w") as f:
        for start in range(0, len(smiles), chunk):
            batch, flat = [], []
            for smi in smiles[start:start + chunk]:
                try:
                    q0, ens = _ensemble(enumerator, template, smi)
                    batch.append((smi, q0, ens))
                    flat += [(len(batch) - 1, q, m) for q, ms in ens.items() for m in ms]
                except Exception as exc:
                    f.write(json.dumps({"smiles": smi, "error": f"enumeration: {exc}"[:200]}) + "\n")
            try:
                g = predict_standard_free_energies([Chem.MolFromSmiles(m) for _, _, m in flat], config=cfg)
                g = g.sort_values("molecule_index")["standard_free_energy"].to_numpy()
            except Exception as exc:
                for smi, _, _ in batch:
                    f.write(json.dumps({"smiles": smi, "error": f"free energy: {exc}"[:200]}) + "\n")
                continue
            for b, (smi, q0, ens) in enumerate(batch):
                rows = [(m, q, gg) for (j, q, m), gg in zip(flat, g) if j == b]
                w = np.array([-(gg + q * math.log(10) * (ph - MU)) for _, q, gg in rows])
                pop = np.exp(w - w.max())
                pop /= pop.sum()
                G = {q: -float(np.logaddexp.reduce([-gg for _, qq, gg in rows if qq == q])) for q in {r[1] for r in rows}}
                macro = {}
                if q0 - 1 in G:
                    macro["acid"] = (G[q0 - 1] - G[q0]) / math.log(10) + MU
                if q0 + 1 in G:
                    macro["base"] = (G[q0] - G[q0 + 1]) / math.log(10) + MU
                f.write(json.dumps({"smiles": smi, "kind": "states", "macro": macro,
                                    "states": [[m, float(p)] for (m, _, _), p in zip(rows, pop)]}) + "\n")
            f.flush()
            print(f"  {min(start + chunk, len(smiles))}/{len(smiles)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tool", choices=["pkasolver", "qupkake", "unipka", "molscrub"])
    ap.add_argument("smiles")
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ph-low", type=float, help="molscrub: enumerate over a pH window instead of one pH")
    ap.add_argument("--ph-high", type=float)
    a = ap.parse_args()
    smiles = [l.strip() for l in open(a.smiles) if l.strip()]
    if a.limit:
        smiles = smiles[: a.limit]
    for var in ("PKASOLVER_DIR", "UNIPKA_DIR", "UNIPKA_WEIGHTS", "UNIPKA_CHECKPOINT"):
        if os.environ.get(var):  # absolute: pkasolver workers chdir before importing
            os.environ[var] = os.path.abspath(os.environ[var])
    if a.tool == "unipka":
        run_unipka(smiles, a.out)
        return
    if a.tool == "molscrub":
        run_molscrub(smiles, a.out, procs=a.procs, ph_low=a.ph_low, ph_high=a.ph_high)
        return
    os.environ["PKA_TMP"] = os.path.dirname(os.path.abspath(a.out))  # worker scratch dirs next to the output
    if a.tool == "pkasolver" and not os.path.isdir(os.path.join(os.environ.get("PKASOLVER_DIR", ""), "pkasolver")):
        sys.exit("PKASOLVER_DIR must point to a pkasolver checkout")  # a failing Pool initializer respawns forever
    if a.tool == "pkasolver":
        with Pool(a.procs, initializer=_pkasolver_init) as p, open(a.out, "w") as f:
            for k, rec in enumerate(p.imap(_pkasolver_one, smiles, chunksize=4)):
                f.write(json.dumps(rec) + "\n")
                if k % 200 == 0:
                    print(f"  {k}/{len(smiles)}", flush=True)
        return
    work = tempfile.mkdtemp(prefix="qupkake_", dir=os.path.dirname(os.path.abspath(a.out)))
    items = list(enumerate(smiles))
    chunks = [(c, items[c::a.procs], work) for c in range(a.procs)]
    try:
        with Pool(a.procs) as p, open(a.out, "w") as f:
            for recs in p.imap_unordered(_qupkake_chunk, chunks):
                for rec in recs:
                    f.write(json.dumps(rec) + "\n")
    finally:
        shutil.rmtree(work, ignore_errors=True)  # xtb scratch and per-molecule caches: ~200 MB per 3k molecules


if __name__ == "__main__":
    main()
