from rdkit import Chem

from vsgen.chem import Options, prepare_2d, variant_seed


def smiles_of(smi, **kw):
    variants, err = prepare_2d(0, "x", smi, Options(**kw))
    assert err is None
    return [v.smiles for v in variants]


def test_desalt_keeps_largest_fragment():
    assert smiles_of("CCN.Cl", tautomers=False) == ["CC[NH3+]"]


def test_assigned_stereo_kept_unassigned_enumerated():
    assert smiles_of("CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O") == ["CC(C)Cc1ccc([C@H](C)C(=O)[O-])cc1"]
    assert len(smiles_of("CC(C)Cc1ccc(cc1)C(C)C(=O)O")) == 2


def test_double_bond_stereo_enumerated():
    out = smiles_of("C(=Cc1ccccc1)c1ccccc1")
    assert sorted(out) == sorted(["C(=C/c1ccccc1)/c1ccccc1", "C(=C\\c1ccccc1)/c1ccccc1"])


def test_variant_cap_and_determinism():
    smi = "OCC1OC(O)C(O)C(O)C1O"  # glucose: 32 stereoisomers
    a, b = smiles_of(smi, max_variants=3), smiles_of(smi, max_variants=3)
    assert len(a) == 3 and a == b


def test_implausible_tautomers_rejected():
    assert smiles_of("CC(=O)Oc1ccccc1C(=O)O") == ["CC(=O)Oc1ccccc1C(=O)[O-]"]  # no ester enol
    assert not any("=N" in s for s in smiles_of("CC(=O)Nc1ccc(O)cc1"))  # no amide iminol


def test_plausible_tautomers_kept():
    out = smiles_of("Oc1ccccn1", ph_min=7.0, ph_max=7.0)
    assert {"Oc1ccccn1", "O=c1cccc[nH]1"} <= set(out)


def test_errors_and_filters():
    assert prepare_2d(0, "x", "C1CC")[1] == "unparseable SMILES"
    assert prepare_2d(0, "x", "CC(=O)O[Cu]OC(C)=O", Options(durrant_filters=True))[1] == "no variant passed filters"


def test_seed_depends_on_chemistry_only():
    a = prepare_2d(0, "x", "CCO")[0][0].seed
    b = prepare_2d(7, "y", "CCO")[0][0].seed
    assert a == b == variant_seed(42, Chem.MolToSmiles(Chem.MolFromSmiles("CCO")))


def test_stereo_outranks_alternative_protomers():
    """The variant budget goes to what the input leaves undefined: a molecule
    with an undefined centre and several protomers gets both stereoisomers of
    the dominant protomer before an alternative charge state."""
    variants, _ = prepare_2d(0, "x", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", Options(max_variants=2))
    assert [v.lineage for v in variants] == ["p0t0s0", "p0t0s1"]
    assert {v.stereo_emitted for v in variants} == {2}


def test_stereo_coverage_is_reported():
    few, _ = prepare_2d(0, "x", "OCC1OC(O)C(O)C(O)C1O")  # 5 undefined centres, 32 isomers
    assert few[0].stereo_undefined == 5 and few[0].stereo_total == 32
    assert few[0].stereo_emitted < few[0].stereo_total  # truncated, and says so
    none, _ = prepare_2d(0, "x", "CC(=O)Oc1ccccc1C(=O)O")
    assert none[0].stereo_undefined == 0 and none[0].stereo_emitted == none[0].stereo_total


def test_enumerated_stereoisomers_differ_in_several_elements():
    """A small budget should not be spent on near-identical isomers."""
    from vsgen.chem import _descriptors

    variants, _ = prepare_2d(0, "x", "CC(N)C(O)CC(C)C=CC", Options(max_variants=4))
    codes = [_descriptors(Chem.MolFromSmiles(v.smiles)) for v in variants]
    assert len(codes) == 4
    assert max(sum(a != b for a, b in zip(codes[0], c)) for c in codes[1:]) >= 3  # not all one-centre flips


def test_ammonium_nitrogen_is_not_a_stereocentre():
    """Proton exchange inverts R3NH+, so its R/S is not a stereoisomer; real N stereo stays."""
    assert smiles_of("CCN(C)Cc1ccccc1", tautomers=False) == ["CC[NH+](C)Cc1ccccc1", "CCN(C)Cc1ccccc1"]
    assert smiles_of("CN1CCC(CC1)c1ccccc1") == ["C[NH+]1CCC(c2ccccc2)CC1"]  # no induced C4 cis/trans
    both = smiles_of("CCCN(CC)C(C)c1ccccc1", tautomers=False)  # the carbon centre is still enumerated
    assert sum("[NH+]" in s for s in both) == 2 and not any("[N@" in s for s in both)
    assert any("[N@" in s for s in smiles_of("CN1CC1C"))  # aziridine N inverts slowly: kept
