from rdkit import Chem

from cgparam.core import get_ring_atoms, get_smarts_matches, mapping


def test_mapping_terminates_for_bridged_ring_without_r1_atoms():
    mol = Chem.MolFromSmiles(
        "C=C[C@H]1C[NH+]2CC[C@H]1C[C@@H]2[C@@H](OC(=O)OCC)"
        "c1ccnc2ccc(OC)cc12"
    )
    matched_maps, _ = get_smarts_matches(mol)

    _, beads, _, _ = mapping(mol, get_ring_atoms(mol), matched_maps, n_iter=3)

    mapped_atoms = [atom for bead in beads for atom in bead]
    assert sorted(mapped_atoms) == list(range(mol.GetNumAtoms()))
