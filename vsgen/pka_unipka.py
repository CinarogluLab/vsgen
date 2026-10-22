"""Protonation microstates from Uni-pKa (optional backend).

Uni-pKa (Luo et al., JACS Au 2024) scores the standard free energy of each
microstate of a protonation ensemble, so populations follow from the ensemble
rather than from per-site pKa. It is ~100x the cost of the default MolGpKa
path (a 190 MB model and 11 conformers per microstate), so vsgen does not call
it inside its workers: `vsgen-protomers` writes a cache of microstates once,
and `vsgen --protomers-from` consumes it. The cache is JSONL, one record per
input, appended as batches finish, so a run can be resumed or extended.

Needs `pip install "vsgen[unipka]"` and a Uni-pKa checkpoint, which is not
redistributed here (see README): pass --checkpoint or set UNIPKA_CHECKPOINT.
Uni-pKa's own ensemble already covers tautomer-like microstates, so the
cached states replace both the protomer and the tautomer stage.
"""

import hashlib
import json
import math
import os

import numpy as np
from rdkit import Chem, RDLogger

from vsgen.chem import canon, desalt

RDLogger.DisableLog("rdApp.*")
TEMPLATE = os.path.join(os.path.dirname(__file__), "data", "unipka_smarts_pattern.tsv")
MU = 6.504894871171601  # Uni-pKa's pKa target normalization (its unimol_pka.py, dwar_8228 mean)
CHARGE_SPAN = 2  # enumerate microstates up to +/- this many protons from the input


def _enumerator():
    from vsgen.third_party import unipka_enumerator as enum

    return enum, enum.read_template(TEMPLATE)


def microstates(smiles, span=CHARGE_SPAN):
    """{net charge: [microstate SMILES]} around the input's charge."""
    enum, template = _enumerator()
    q0 = Chem.GetFormalCharge(Chem.MolFromSmiles(smiles))
    ens, same = {}, [smiles]
    same, lower = enum.enumerate_template(same, *template, maxiter=10, mode="A")
    if lower:
        ens[q0 - 1] = lower
        for q in range(q0 - 2, q0 - span - 1, -1):
            if q + 1 in ens:
                _, deeper = enum.enumerate_template(ens[q + 1], *template, maxiter=10, mode="A")
                if deeper:
                    ens[q] = deeper
    upper, same = enum.enumerate_template(same, *template, maxiter=10, mode="B")
    if upper:
        ens[q0 + 1] = upper
        for q in range(q0 + 2, q0 + span + 1):
            if q - 1 in ens:
                higher, _ = enum.enumerate_template(ens[q - 1], *template, maxiter=10, mode="B")
                if higher:
                    ens[q] = higher
    ens[q0] = same
    return q0, ens


def _populations(rows, q0, ph):
    """Boltzmann populations of (smiles, charge, free energy) rows at a pH, and
    the macro pKa of the two adjacent transitions."""
    weight = np.array([-(g + q * math.log(10) * (ph - MU)) for _, q, g in rows])
    pop = np.exp(weight - weight.max())
    pop /= pop.sum()
    agg = {q: -float(np.logaddexp.reduce([-g for _, qq, g in rows if qq == q])) for q in {r[1] for r in rows}}
    macro = {}
    if q0 - 1 in agg:
        macro["acid"] = (agg[q0 - 1] - agg[q0]) / math.log(10) + MU
    if q0 + 1 in agg:
        macro["base"] = (agg[q0] - agg[q0 + 1]) / math.log(10) + MU
    return pop, macro


def build_cache(inputs, out_path, checkpoint, ph=7.4, batch=40, gpu=False, min_population=0.05,
                max_states=5, log=None):
    """Write/extend a protomer cache for `inputs` [(name, smiles)].

    Records: {"smiles": canonical desalted input, "states": [[smiles, population]],
    "macro_pka": {...}, "meta": {...}} or {"smiles", "error"}. Inputs already in
    the cache are skipped, so interrupted runs resume.
    """
    from unipkainfer import UnipkaFreeEnergyConfig, predict_standard_free_energies

    meta = {"source": "unipka", "checkpoint_sha256": _sha256(checkpoint)[:16], "template": os.path.basename(TEMPLATE),
            "ph": ph, "charge_span": CHARGE_SPAN, "min_population": min_population, "max_states": max_states}
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                # Only entries built with this configuration count as done: changing the
                # pH, checkpoint or caps must not leave stale states in the cache.
                if rec.get("meta") == meta and not rec.get("error"):
                    done.add(rec["smiles"])
                else:
                    done.discard(rec["smiles"])
    _patch_unipkainfer(checkpoint)
    cfg = UnipkaFreeEnergyConfig(model_dir=os.path.dirname(checkpoint) or ".", gpu=gpu)
    todo, written = [], 0
    for name, smiles in inputs:
        mol = Chem.MolFromSmiles(smiles)
        key = canon(desalt(mol)) if mol is not None else None
        if key is None or key in done:
            continue
        done.add(key)
        todo.append(key)
    with open(out_path, "a") as f:
        for start in range(0, len(todo), batch):
            chunk, flat = [], []
            for key in todo[start:start + batch]:
                try:
                    q0, ens = microstates(key)
                except Exception as exc:
                    f.write(json.dumps({"smiles": key, "error": f"enumeration: {exc}"[:200]}) + "\n")
                    continue
                chunk.append((key, q0, ens))
                flat += [(len(chunk) - 1, q, m) for q, ms in ens.items() for m in ms]
            if not chunk:
                continue
            try:
                energies = predict_standard_free_energies([Chem.MolFromSmiles(m) for _, _, m in flat], config=cfg)
                energies = energies.sort_values("molecule_index")["standard_free_energy"].to_numpy()
            except Exception as exc:
                for key, _, _ in chunk:
                    f.write(json.dumps({"smiles": key, "error": f"free energy: {exc}"[:200]}) + "\n")
                continue
            for i, (key, q0, _) in enumerate(chunk):
                rows = [(m, q, g) for (j, q, m), g in zip(flat, energies) if j == i]
                pop, macro = _populations(rows, q0, ph)
                states = sorted(((m, float(p)) for (m, _, _), p in zip(rows, pop)), key=lambda x: -x[1])
                states = [[m, p] for m, p in states if p >= min_population][:max_states] or [[states[0][0], 1.0]]
                total = sum(p for _, p in states)  # renormalized over what is kept, so they sum to 1
                states = [[m, p / total] for m, p in states]
                f.write(json.dumps({"smiles": key, "states": states, "macro_pka": macro, "meta": meta,
                                    "device": "cuda" if gpu else "cpu"}) + "\n")
                written += 1
            f.flush()
            if log:
                print(f"  {min(start + batch, len(todo))}/{len(todo)} inputs", file=log, flush=True)
    return {"inputs": len(inputs), "new": written, "cached": len(done) - written, "cache": out_path}


def _patch_unipkainfer(checkpoint):
    """Two fixes to unipkainfer 0.2.0, applied at runtime.

    1. Point it at one checkpoint file instead of downloading folds by name.
    2. Zero-pad its LMDB record keys. It writes keys "0".."N" but reads them
       back in cursor (lexicographic) order, so with more than 10 microstates
       in a call the free energies come back attached to the wrong molecules
       (e.g. imidazole's base pKa 4.0 instead of 7.2).
    """
    import pickle
    from pathlib import Path

    import lmdb
    from unipkainfer.pka_predictor import free_energy as fe

    fe.ensure_checkpoint = lambda model_dir, fold, **kw: Path(checkpoint)

    def write_lmdb(smiles, task_name, processed_dir, cfg):
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


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_cache(path):
    """{canonical desalted input SMILES: [state SMILES, best population first]},
    with None where the cache records a failure, so callers can tell a failed
    input from one that was never cached. A later record replaces an earlier
    one, which is how a rebuild with new settings supersedes stale entries."""
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[rec["smiles"]] = None if rec.get("error") else [s for s, _ in rec["states"]]
    return out
