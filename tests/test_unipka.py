import importlib.util
import io
import json
import math
import os

import pytest
from rdkit import Chem

from vsgen import pka_unipka
from vsgen.chem import Options, cache_key, prepare_2d

CHECKPOINT = os.environ.get("UNIPKA_CHECKPOINT", "")


def _unipkainfer():
    return importlib.util.find_spec("unipkainfer") is not None


def test_cache_key_is_the_desalted_canonical_input():
    assert cache_key("CCN.Cl") == "CCN"
    assert cache_key("c1ccccc1O") == cache_key("Oc1ccccc1")
    assert cache_key("C1CC") == ""  # unparseable


def test_cached_states_replace_protomers_and_tautomers():
    states = ["CC(=O)Oc1ccccc1C(=O)[O-]", "CC(=O)Oc1ccccc1C(=O)O"]
    variants, err = prepare_2d(0, "aspirin", "CC(=O)Oc1ccccc1C(=O)O", Options(), states)
    assert err is None
    assert [v.smiles for v in variants] == states  # in cache order, no tautomers added
    assert [v.lineage for v in variants] == ["p0t0s0", "p1t0s0"]


def test_cached_states_still_get_stereo_enumeration():
    variants, _ = prepare_2d(0, "x", "CC(C)NCC(O)COc1cccc2ccccc12",
                             Options(), ["CC(C)[NH2+]CC(O)COc1cccc2ccccc12"])
    assert len(variants) == 2 and {v.lineage for v in variants} == {"p0t0s0", "p0t0s1"}


def test_load_cache_keeps_order_and_marks_failures(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(
        json.dumps({"smiles": "CCO", "states": [["CCO", 0.9], ["CC[O-]", 0.1]]}) + "\n"
        + json.dumps({"smiles": "CCN", "error": "enumeration: boom"}) + "\n"
        + json.dumps({"smiles": "CCS", "states": [["CCS", 1.0]]}) + "\n"
        + json.dumps({"smiles": "CCS", "states": [["CC[S-]", 1.0]]}) + "\n")  # rebuilt entry
    cache = pka_unipka.load_cache(str(path))
    assert cache == {"CCO": ["CCO", "CC[O-]"], "CCN": None, "CCS": ["CC[S-]"]}


def test_uncached_and_failed_inputs_do_not_fall_back_silently(tmp_path):
    """A partial cache must not quietly mix Uni-pKa and the default model."""
    from vsgen.run import RunConfig, run

    cache = tmp_path / "c.jsonl"
    cache.write_text(json.dumps({"smiles": "CC(=O)Oc1ccccc1C(=O)O",
                                 "states": [["CC(=O)Oc1ccccc1C(=O)[O-]", 1.0]]}) + "\n"
                     + json.dumps({"smiles": "CCO", "error": "free energy: boom"}) + "\n")
    src = tmp_path / "in.smi"
    src.write_text("CC(=O)Oc1ccccc1C(=O)O aspirin\nCCO ethanol\nCCN ethylamine\n")
    out = tmp_path / "out.sdf"
    cfg = RunConfig(gpu="off", procs=2, protomers_from=str(cache))
    summary = run(str(src), str(out), cfg, log=io.StringIO())
    assert summary["inputs_with_models"] == 1  # only the cached one
    failures = (tmp_path / "out_failed.tsv").read_text()
    assert failures.count("protomer cache") == 2  # the failed entry and the uncached input

    cfg.protomers_fallback = True
    out2 = tmp_path / "out2.sdf"
    assert run(str(src), str(out2), cfg, log=io.StringIO())["inputs_with_models"] == 3


def test_retained_populations_are_renormalized():
    rows = [("A", 0, 0.0), ("B", 0, 2.0), ("C", -1, 9.0)]  # C is far too minor to keep
    pop, _ = pka_unipka._populations(rows, 0, 7.4)
    kept = [(m, p) for (m, _, _), p in zip(rows, pop) if p >= 0.05]
    total = sum(p for _, p in kept)
    assert len(kept) < len(rows) and sum(p / total for _, p in kept) == pytest.approx(1.0)


def test_populations_follow_henderson_hasselbalch():
    # one site, free energies chosen so the macro pKa is exactly MU: at pH = MU
    # the neutral and anionic forms are equally populated.
    rows = [("A", 0, 0.0), ("A-", -1, 0.0)]
    pop, macro = pka_unipka._populations(rows, 0, pka_unipka.MU)
    assert macro["acid"] == pytest.approx(pka_unipka.MU)
    assert pop[0] == pytest.approx(0.5) and pop[1] == pytest.approx(0.5)
    # two pH units above that pKa the anion is ~99% of the ensemble
    pop, _ = pka_unipka._populations(rows, 0, pka_unipka.MU + 2)
    assert pop[1] == pytest.approx(100 / 101, abs=1e-3)


def test_microstates_cover_both_directions():
    q0, ens = pka_unipka.microstates("NCC(=O)O")  # glycine: can lose H+ and gain H+
    assert q0 == 0 and set(ens) >= {-1, 0, 1}
    assert any("[O-]" in s for s in ens[-1]) and any("[NH3+]" in s for s in ens[1])


@pytest.mark.skipif(not CHECKPOINT, reason="needs UNIPKA_CHECKPOINT and vsgen[unipka]")
def test_backend_reproduces_known_pka(tmp_path):
    """Chlorophenol pKa 9.4, imidazole 7.0: a batch >10 microstates also checks
    that the LMDB key fix keeps energies attached to the right microstate."""
    inputs = [("clphenol", "Oc1ccc(Cl)cc1"), ("imidazole", "c1c[nH]cn1"),
              ("propranolol", "CC(C)NCC(O)COc1cccc2ccccc12"), ("glycine", "NCC(=O)O")]
    cache_path = tmp_path / "cache.jsonl"
    pka_unipka.build_cache(inputs, str(cache_path), CHECKPOINT)
    recs = {json.loads(l)["smiles"]: json.loads(l) for l in open(cache_path)}
    assert recs[cache_key("Oc1ccc(Cl)cc1")]["macro_pka"]["acid"] == pytest.approx(9.4, abs=1.0)
    assert recs[cache_key("c1c[nH]cn1")]["macro_pka"]["base"] == pytest.approx(7.0, abs=1.0)
    prop = recs[cache_key("CC(C)NCC(O)COc1cccc2ccccc12")]
    assert "[NH2+]" in prop["states"][0][0]  # amine protonated at pH 7.4 (pKa 9.5)
    assert math.isclose(sum(p for _, p in prop["states"]), 1.0, abs_tol=1e-6)  # renormalized


@pytest.mark.skipif(not CHECKPOINT, reason="needs UNIPKA_CHECKPOINT and vsgen[unipka]")
def test_cache_rebuilds_when_the_configuration_changes(tmp_path):
    cache = tmp_path / "c.jsonl"
    inputs = [("clphenol", "Oc1ccc(Cl)cc1")]
    assert pka_unipka.build_cache(inputs, str(cache), CHECKPOINT, ph=7.4)["new"] == 1
    assert pka_unipka.build_cache(inputs, str(cache), CHECKPOINT, ph=7.4)["new"] == 0  # resumes
    assert pka_unipka.build_cache(inputs, str(cache), CHECKPOINT, ph=12.0)["new"] == 1  # pH changed
    assert pka_unipka.load_cache(str(cache))[cache_key("Oc1ccc(Cl)cc1")][0] == "[O-]c1ccc(Cl)cc1"


@pytest.mark.skipif(not _unipkainfer(), reason="needs vsgen[unipka]")
def test_patched_unipkainfer_internals_still_exist():
    """vsgen patches unipkainfer internals; catch an API change before it
    silently produces scrambled energies (see _patch_unipkainfer)."""
    from unipkainfer.pka_predictor import free_energy as fe

    for attr in ("ensure_checkpoint", "_write_free_energy_lmdb", "_smiles_to_free_energy_record",
                 "_UNIPKA_VALID_SUBSET"):
        assert hasattr(fe, attr), f"unipkainfer no longer has {attr}"
