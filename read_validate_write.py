#!/usr/bin/env python
import csv

from rdkit import Chem
from tqdm import tqdm

from util import is_valid_smiles


def read_validate_write(path: str) -> None:
    """Validates the Pubchem database of SMILES strings."""

    data = []
    with open(path) as file, tqdm(total=115_371_812, desc="Validating CID-SMILES...") as pbar:
        while line := file.readline():
            smiles = line.split()[1]
            assert is_valid_smiles(smiles)
            data.append(smiles)
            pbar.update(1)

    with open("SMILES.csv", "w", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["text"])
        for smiles in data:
            writer.writerow([smiles])


if __name__ == "__main__":
    read_validate_write("CID-SMILES")
