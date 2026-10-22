import pytest
from rdkit import Chem

from vsgen import pka
from vsgen.chem import Options, canon, protomers

# Outputs of the original PyTorch MolGpKa (github.com/Xundrug/MolGpKa) on these
# molecules: acid sites keyed by heavy atom, base sites by atom.
REFERENCE = {
    "Oc1ccccc1": ({0: 10.1182}, {}),
    "NS(=O)(=O)c1ccccc1": ({0: 10.139}, {0: 4.1412}),
    "C1CCNCC1": ({}, {3: 10.6107}),
    "c1c[nH]cn1": ({2: 13.8795}, {4: 6.7844}),
    "CC(C)NCC(O)COc1cccc2ccccc12": ({6: 13.7716}, {3: 9.0533}),
    "NCC(=O)O": ({4: 2.0931}, {0: 9.6356}),
}


@pytest.mark.parametrize("smi", REFERENCE)
def test_numpy_port_matches_pytorch_original(smi):
    _, acids, bases = pka.predict(Chem.MolFromSmiles(smi))
    ref_acid, ref_base = REFERENCE[smi]
    assert acids.keys() == ref_acid.keys() and bases.keys() == ref_base.keys()
    for got, ref in ((acids, ref_acid), (bases, ref_base)):
        for k in ref:
            assert got[k] == pytest.approx(ref[k], abs=1e-3)


def states(smi, **kw):
    return {canon(m) for m in protomers(smi, Options(**kw))}


@pytest.mark.parametrize("smi, expected", [
    ("Oc1ccccc1", {"Oc1ccccc1"}),  # phenol pKa 10: neutral (Dimorphite-DL also emits phenolate)
    ("NS(=O)(=O)c1ccccc1", {"NS(=O)(=O)c1ccccc1"}),  # aryl sulfonamide pKa 10: neutral
    ("CC(=O)Nc1ccccc1", {"CC(=O)Nc1ccccc1"}),  # amide: never deprotonated
    ("CC(=O)Oc1ccccc1C(=O)O", {"CC(=O)Oc1ccccc1C(=O)[O-]"}),  # carboxylic acid: anion
    ("CN1CCNCC1", {"C[NH+]1CCNCC1"}),  # second piperazine N pushed below the window
    ("CCN", {"CC[NH3+]"}),  # tiny molecule: Dimorphite-DL fallback
])
def test_dominant_states(smi, expected):
    assert states(smi) == expected


def test_zwitterion_is_emitted():
    assert "[NH3+][C@@H](Cc1ccccc1)C(=O)[O-]" in states("N[C@@H](Cc1ccccc1)C(=O)O")  # phenylalanine


def test_sulfonamide_not_deprotonated():  # Astex 1HNN ligand; Dimorphite-DL emitted the [NH-] form
    assert not any("[NH-]" in s for s in states("NS(=O)(=O)c1ccc2c(c1)CCNC2"))


def test_near_neutral_sites_give_both_forms():
    assert states("c1c[nH]cn1") == {"c1c[nH]cn1", "c1c[nH+]c[nH]1"}  # imidazole pKa 7.0


def test_dimorphite_still_available():  # 4-chlorophenol, pKa 9.4: only Dimorphite-DL emits the phenolate
    assert "[O-]c1ccc(Cl)cc1" in states("Oc1ccc(Cl)cc1", pka_model="dimorphite")
    assert states("Oc1ccc(Cl)cc1") == {"Oc1ccc(Cl)cc1"}
