#!/usr/bin/env python
from rdkit import Chem

from tqdm import tqdm


def is_valid(smiles: str) -> bool:
    """Checks whether the given SMILES string is syntactically valid.

    See: https://github.com/rdkit/rdkit/issues/2430
    """

    return False if Chem.MolFromSmiles(smiles, sanitize=False) is None else True


def read_and_validate(path: str, size_limit: int = 100_000) -> list[str]:
    """Validates the Pubchem database of SMILES strings."""

    data = []
    with open(path, encoding="utf-8") as file, tqdm(total=size_limit, desc="Validating CID-SMILES") as pbar:
        count = 0
        while (line := file.readline()) and count < size_limit:
            smiles = line.split()[1]
            assert is_valid(smiles)
            data.append(smiles)
            count += 1
            pbar.update(1)

    return data


if __name__ == "__main__":
    _ = read_and_validate("data/CID-SMILES")