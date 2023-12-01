#!/usr/bin/env python
import matplotlib.pyplot as plt

from collections import defaultdict

from tqdm import tqdm


plt.style.use("tableau-colorblind10")


def gen_smiles_dist(path: str) -> None:
    """Generates a distribution of SMILES strings across length ranges."""

    data = defaultdict(int)
    with open(path) as file, tqdm(total=115_371_812, desc="Generating a diagram") as pbar:
        while line := file.readline():
            smiles = line.split()[1]
            found = False
            for length in range(10, 101, 10):
                if len(smiles) < length:
                    data[f"< {length}"] += 1
                    found = True
                    break

            if not found:
                data[">= 100"] += 1

            pbar.update(1)

        data = dict(sorted(data.items()))

        plt.bar(data.keys(), data.values())
        plt.title("Distribution of SMILES Strings Across Length Ranges")
        plt.xlabel("Length Range")
        plt.ylabel("SMILES Strings")
        plt.savefig("smiles_dist.png")


if __name__ == "__main__":
    gen_smiles_dist("CID-SMILES")
