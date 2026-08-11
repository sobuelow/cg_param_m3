import warnings

import numpy as np
import pytest
from rdkit import Chem

from cgparam.core import (
    construct_vs,
    get_ring_atoms,
    get_smi,
    get_smarts_matches,
    mapping,
)


BRL1_156 = (
    "CC[C@H](C)[C@H]1O[C@]2(CC[C@@H]1C)C[C@@H]1C[C@@H](C/C=C(\\C)"
    "[C@@H](O[C@H]3C[C@H](OC)[C@@H](O[C@H]4C[C@H](OC)[C@@H](O)[C@H](C)O4)"
    "[C@H](C)O3)[C@@H](C)/C=C/C=C3\\CO[C@@H]4[C@H](O)C(C)=C[C@@H](C(=O)O1)"
    "[C@]34O)O2"
)


def assert_complete_partition(mol, beads):
    mapped_atoms = [atom for bead in beads for atom in bead]
    assert sorted(mapped_atoms) == list(range(mol.GetNumAtoms()))
    assert len(mapped_atoms) == len(set(mapped_atoms))


def test_mapping_terminates_for_bridged_ring_without_r1_atoms():
    mol = Chem.MolFromSmiles(
        "C=C[C@H]1C[NH+]2CC[C@H]1C[C@@H]2[C@@H](OC(=O)OCC)"
        "c1ccnc2ccc(OC)cc12"
    )
    matched_maps, _ = get_smarts_matches(mol)

    _, beads, _, _ = mapping(mol, get_ring_atoms(mol), matched_maps, n_iter=3)

    assert_complete_partition(mol, beads)


def test_fragment_smiles_ignores_stereochemistry_crossing_bead_boundary():
    mol = Chem.MolFromSmiles(BRL1_156)

    bead_smi, _, _ = get_smi([46, 44, 45], mol)

    assert bead_smi == "C=CC"


def test_cobalt_symbol_is_not_mistaken_for_aromatic_oxygen():
    mol = Chem.MolFromSmiles("N#[C][Co+]")

    bead_smi, ring_size, frag_size = get_smi([0, 1, 2], mol)

    assert bead_smi == "N#[C][Co+]"
    assert ring_size == 0
    assert frag_size == 0


def test_virtual_site_uses_a_triangle_that_contains_it():
    coords = np.array([
        [1.3355318553000226, -0.5071047045384752],
        [0.2916803563558019, -0.03379043476737232],
        [-0.44114520276218416, -0.5079609703372195],
        [0.6300825914455861, -0.3018676045339098],
        [-0.15144364817129202, 0.022221557194479512],
        [1.1765083202520015, 0.6805109746331383],
        [0.3826002677279928, -0.5635713933532417],
        [-1.3819687200064654, 0.949529931735302],
        [0.21430530410227347, 0.010280767627900755],
    ])
    real_sites = [6, 0, 5, 7, 2]

    weights = construct_vs(8, real_sites, coords, list(range(len(coords))))

    reconstructed = sum(weight * coords[site] for site, weight in weights.items())
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(weight >= -1.0e-12 for weight in weights.values())
    assert reconstructed == pytest.approx(coords[8])


def test_ring_fragment_indices_are_recovered_from_atom_maps():
    mol = Chem.MolFromSmiles(BRL1_156)
    matched_maps, _ = get_smarts_matches(mol)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, beads, _, _ = mapping(
            mol, get_ring_atoms(mol), matched_maps, n_iter=3
        )

    assert_complete_partition(mol, beads)
    assert not any("divide by zero" in str(warning.message) for warning in caught)
    assert not any("invalid value" in str(warning.message) for warning in caught)


def test_smarts_match_is_rejected_when_it_strands_an_atom():
    mol = Chem.MolFromSmiles("CC[N+](C)(C)C")

    matched_maps, matched_beads = get_smarts_matches(mol)

    assert matched_maps == []
    assert matched_beads == []


@pytest.mark.parametrize(
    "smiles",
    [
        "CC[C@@]1(O)C(=O)OCc2c1cc1n(c2=O)Cc2cc3ccccc3nc2-1",
        (
            "CCCCCC[C@H]1C(=O)O[C@H](C)C(NC(=O)c2cccc(NC=O)c2O)C(=O)O"
            "[C@@H](C)[C@@H]1OC(=O)CC(C)C"
        ),
        "C[N+]1(C)CCOC(O)(c2ccc(-c3ccc(C4(O)C[N+](C)(C)CCO4)cc3)cc2)C1",
    ],
)
def test_fixed_smarts_beads_remain_exact_during_ring_mapping(smiles):
    mol = Chem.MolFromSmiles(smiles)
    matched_maps, _ = get_smarts_matches(mol)

    _, beads, _, _ = mapping(mol, get_ring_atoms(mol), matched_maps, n_iter=3)

    assert_complete_partition(mol, beads)
    bead_sets = {frozenset(bead) for bead in beads}
    assert all(frozenset(match) in bead_sets for match in matched_maps)
