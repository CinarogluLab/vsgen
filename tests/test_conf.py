import time

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolAlign

from vsgen.conf import Conf3D, build_cpu, stereo_preserved


def test_chiral_molecule_gets_3d_with_stereo():
    smi = "C[C@@H](N)C(=O)[O-]"
    models, err = build_cpu(smi, 1)
    assert err is None and models
    mol, energy, kind = models[0]
    assert kind == "best" and energy is not None and np.isfinite(energy)
    assert mol.GetConformer().Is3D() and mol.GetNumAtoms() > Chem.MolFromSmiles(smi).GetNumAtoms()
    assert stereo_preserved(Chem.MolFromSmiles(smi), mol)


def test_same_seed_same_coordinates():
    a = build_cpu("CCOc1ccccc1C(=O)N", 7)[0][0][0].GetConformer().GetPositions()
    b = build_cpu("CCOc1ccccc1C(=O)N", 7)[0][0][0].GetConformer().GetPositions()
    assert np.allclose(a, b)


def test_impossible_stereoisomer_fails_fast():
    # Inverted camphor-type bridgehead: cannot be embedded. Uncapped ETKDG spent ~40 s here, 15 capped starts ~23 s.
    t = time.perf_counter()
    models, err = build_cpu("CCCCCN(CCCCC)CC(=O)O[C@@H]1C[C@@H]2CC[C@]1(C)C2(C)C", 1)
    assert not models and "embedding failed" in err
    assert time.perf_counter() - t < 5


def test_ring_alternates_differ():
    models, _ = build_cpu("O[C@H]1CC[C@@H](N)CC1", 3, Conf3D(thorough=True, ring_confs=3))
    assert [kind for _, _, kind in models] == ["best"] + ["ring_alt"] * (len(models) - 1)
    ref = Chem.MolFromSmiles("O[C@H]1CC[C@@H](N)CC1")  # keep alive: RingInfo borrows from it
    ring = list(ref.GetRingInfo().AtomRings()[0])
    best = models[0][0]
    for alt, _, _ in models[1:]:
        assert rdMolAlign.AlignMol(Chem.Mol(alt), best, atomMap=[(i, i) for i in ring]) > 0.3


@pytest.mark.parametrize("smi", ["CC(=O)O[Cu]OC(C)=O", "C[Si](C)(C)OC"])
def test_unusual_atoms_do_not_crash(smi):
    models, err = build_cpu(smi, 1)
    assert models or err


def test_mmff94s_default_with_uff_fallback():
    models, _ = build_cpu("CC(C)[NH2+]CC(O)COc1cccc2ccccc12", 1)
    assert models[0][0].GetProp("force_field") == "MMFF94s"
    boronic, _ = build_cpu("OB(O)c1ccccc1", 1)  # MMFF has no boron parameters
    assert boronic[0][0].GetProp("force_field") == "UFF"
    assert build_cpu("CCO", 1, Conf3D(force_field="uff"))[0][0][0].GetProp("force_field") == "UFF"


def test_mmff_dielectric_is_distance_dependent():
    """eps = 4r, not a constant 4 (RDKit's model 1 is constant, 2 distance-dependent)."""
    from rdkit.Chem import AllChem
    from vsgen.conf import mmff_props

    mol = build_cpu("[NH3+]CCCCCC(=O)[O-]", 1)[0][0][0]
    const = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    const.SetMMFFDielectricConstant(4.0)
    e = lambda p: AllChem.MMFFGetMoleculeForceField(mol, p).CalcEnergy()
    assert abs(e(mmff_props(mol)) - e(const)) > 1.0


def test_unminimized_models_are_flagged():
    models, _ = build_cpu("CC(=O)O[Cu]OC(C)=O", 1)
    assert all(m.GetProp("force_field") in ("MMFF94s", "UFF", "none") for m, _, _ in models)
