#!/usr/bin/env python
import matplotlib.pyplot as plt

from tqdm import tqdm


plt.style.use("tableau-colorblind10")


def gen_smiles_dist(path: str) -> None:
    """Generates a distribution of SMILES strings across length ranges."""

    data = {"< 10": 0, "< 20": 0, "< 30": 0, "< 40": 0, "< 50": 0, "< 60": 0, "< 70": 0, ">= 70": 0}
    with open(path) as file, tqdm(total=115_371_812, desc="Generating a diagram") as pbar:
        while line := file.readline():
            smiles = line.split()[1]
            if len(smiles) < 10:
                data["< 10"] += 1
            elif len(smiles) < 20:
                data["< 20"] += 1
            elif len(smiles) < 30:
                data["< 30"] += 1
            elif len(smiles) < 40:
                data["< 40"] += 1
            elif len(smiles) < 50:
                data["< 50"] += 1
            elif len(smiles) < 60:
                data["< 60"] += 1
            elif len(smiles) < 70:
                data["< 70"] += 1
            elif len(smiles) >= 70:
                data[">= 70"] += 1
            pbar.update(1)

        plt.bar(data.keys(), data.values())
        plt.title("Distribution of SMILES Strings Across Length Ranges")
        plt.xlabel("Length Range")
        plt.ylabel("SMILES Strings")
        plt.savefig("smiles_dist.png")


if __name__ == "__main__":
    gen_smiles_dist("CID-SMILES")
