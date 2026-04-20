import argparse
import logging
from pathlib import Path

from rdkit import Chem

from .core import CGParam

logger = logging.getLogger(__name__)

    # if args.file:
    #     with Chem.SDMolSupplier(args.file) as suppl:
    #         mols_list = [x for x in suppl if x is not None]
    #     smiles_list = [Chem.MolToSmiles(mol,canonical=True) for mol in mols_list]
    #         # mol = ms[0]
    #         # smi = Chem.MolToSmiles(mol,canonical=False)
    # else:

def main():

    print_welcome()

    parser = argparse.ArgumentParser(description="CG parameter mapping (M3)")
    parser.add_argument('-s','--smiles',help='SMILES code of the molecule.',required=False)
    parser.add_argument('-n','--name',help='Name of the molecule.',default='molecule')
    parser.add_argument('--path_out',help='Output folder.',nargs='?',type=str,default='.')
    parser.add_argument("--iter", type=int, default=3)
    parser.add_argument('-t',help='Tuning: Setting to enable tuned bead parameterisation. This uses the log Kow of neighbouring beads as well as the log Kow of the bead in question when parameterising a bead. Developed focusing on diesters, for an upcoming publication.',action='store_true')
    parser.add_argument("-v", "--verbose", action="count", default=0)
    
    args = parser.parse_args()

    configure_logging(args.verbose)

    path_out = Path(args.path_out)

    mol = prep_mol(args.smiles)

    cgparam = CGParam(
        name=args.name,
        mol = mol,
        n_iter = args.iter,
        tune = args.t,
        path_out = path_out,
    )

    cgparam.run_pipeline()

    logger.debug("")
    logger.debug("All done. Thanks for using cg_param!")

def prep_mol(smiles):
    if not smiles:
        raise FileNotFoundError('No SMILES string provided')
    else:
        mol = Chem.MolFromSmiles(smiles)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    return mol

def configure_logging(verbose):
    # Configure logging level
    level = logging.WARNING  # default
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(levelname)s:%(name)s:%(message)s"
    )
    return None

#Start of script introduction
def print_welcome():
    logger.info("")
    logger.info("Thanks for using cg_kmw. Latest version: https://github.com/cgkmw-durham/cg_param_m3")
    logger.info("")
    logger.info("The original version of this script for the Martini 2 forcefield, and a full description of the mapping and") 
    logger.info("parametrisation procedures, can be found in the following paper:")
    logger.info("T.D. Potter, E.L. Barrett and M.A. Miller, Automated Coarse-Grained Mapping Algorithm for the Martini Force Field and") 
    logger.info("Benchmarks for Membrane–Water Partitioning, J. Chem. Theory Comput., 2021, https://doi.org/10.1021/acs.jctc.1c00322.")
    logger.info("")

    logger.debug("Colour coding of dumped arrays: ", "\033[38;5;34m","Atoms ","\033[0;0m", "Vs ", "\033[38;5;128m","Beads","\033[0;0m")
    logger.debug("Atoms Numbered According to smiles, bead mapping is arbitrary")

if __name__ == "__main__":
    main()