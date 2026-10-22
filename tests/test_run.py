import io

import numpy as np
import pytest
from rdkit import Chem

from vsgen.run import RunConfig, _gpu_ok, run

SMILES = """CC(=O)Oc1ccccc1C(=O)O aspirin
not_a_smiles broken
CN1CCC[C@H]1c1cccnc1 nicotine
CCN.Cl ethylamine_hcl
C(=Cc1ccccc1)c1ccccc1 stilbene
"""


def _run(tmp_path, gpu="off", name="out"):
    src = tmp_path / "in.smi"
    src.write_text(SMILES)
    out = tmp_path / f"{name}.sdf"
    summary = run(str(src), str(out), RunConfig(gpu=gpu, procs=2), log=io.StringIO())
    return summary, [m for m in Chem.SDMolSupplier(str(out), removeHs=False)], tmp_path / f"{name}_failed.tsv"


def test_end_to_end_cpu(tmp_path):
    summary, mols, failed = _run(tmp_path)
    assert summary["inputs"] == 5 and summary["inputs_with_models"] == 4
    names = [m.GetProp("_Name") for m in mols]
    assert names == sorted(names, key=["aspirin", "nicotine", "ethylamine_hcl", "stilbene"].index)  # input order
    assert all(m.GetConformer().Is3D() and m.GetProp("backend") == "cpu" for m in mols)
    assert "broken" in failed.read_text()


def test_cpu_runs_are_reproducible(tmp_path):
    _, a, _ = _run(tmp_path, name="a")
    _, b, _ = _run(tmp_path, name="b")
    assert [m.GetProp("SMILES") for m in a] == [m.GetProp("SMILES") for m in b]
    assert all(np.allclose(x.GetConformer().GetPositions(), y.GetConformer().GetPositions()) for x, y in zip(a, b))


@pytest.mark.skipif(not _gpu_ok(), reason="needs nvMolKit and an NVIDIA GPU")
def test_end_to_end_gpu(tmp_path):
    summary, mols, _ = _run(tmp_path, gpu="on")
    assert summary["gpu"] and summary["inputs_with_models"] == 4
    assert {m.GetProp("backend") for m in mols} <= {"cpu", "gpu"} and "gpu" in {m.GetProp("backend") for m in mols}
    assert all(m.GetConformer().Is3D() for m in mols)
