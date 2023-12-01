#!/usr/bin/env python
import util

from datasets import load_dataset


# NOTE: A bit over 10_000_000 to ensure we get the exact number
DATASET_SIZE: int = 10_500_000


def promptify(row: dict[str, str]) -> str:
    try:
        smiles = util.canonicalize_smiles(row["smiles"])
        bitstr = " ".join(list(util.maccs_fingerprint(smiles).ToBitString()))
        prompt = f"{bitstr}\n{smiles}"
        return {"prompt": prompt}
    except:
        return {"prompt": ""}


def main() -> None:
    pubchem_smiles = (
        load_dataset(
            "csv",
            delimiter="\t",
            column_names=["id", "smiles"],
            data_files=["CID-SMILES"],
        )
        .map(promptify)
        .filter(lambda row: 0 < len(row["prompt"]) <= 512)
        .remove_columns(["id", "smiles"])
    )

    pubchem_smiles.to_csv("pubchem_smiles.csv")


if __name__ == "__main__":
    main()
