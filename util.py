import argparse
import csv
import heapq
import json
import pathlib

import tqdm

from collections import defaultdict

from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors

from sentence_transformers import SentenceTransformer


RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    """Parses the command line arguments."""

    parser = argparse.ArgumentParser("Automatic captioning of molecules")

    parser.add_argument("--train", type=str, help="path to training dataset", required=True)
    parser.add_argument("--test", type=str, help="path to testing dataset", required=True)
    parser.add_argument("--val", type=str, help="path to validation dataset", required=True)

    return parser.parse_args()


def morgan_fingerprint(smile: str) -> DataStructs.cDataStructs.ExplicitBitVect:
    """Generates Morgan fingerprints using the given SMILES string."""

    mol = Chem.MolFromSmiles(smile)
    mor = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2)

    return mor


class ChEBI20:
    """ChEBI-20 dataset.

    Paper: https://aclanthology.org/2022.emnlp-main.26/
    Code: https://github.com/blender-nlp/MolT5
    Data: https://github.com/blender-nlp/MolT5/tree/main/ChEBI-20_data
    """

    DType = dict[str, list[str]]

    def __init__(self, train_path: str, test_path: str, val_path: str) -> None:
        """Reads and validates ChEBI-20 data files."""

        # Train, test, and val sizes must be 26_407, 3_300, and 3_301, respectively
        self.train = self._read(train_path)
        assert len(self.train) == 26_407

        self.test = self._read(test_path)
        assert len(self.test) == 3_300

        self.val = self._read(val_path)
        assert len(self.val) == 3_301

    def _read(self, path: str) -> DType:
        """Reads ChEBI-20 data files."""

        with open(path) as file:
            data = list(csv.DictReader(file, delimiter="\t"))

        return data

    def _generate_morgan_fingerprints(self, data: DType) -> None:
        """Generates Morgan fingerprints."""

        for item in tqdm.tqdm(data, desc="Generating Morgan fingerprints"):
            item["MORGAN"] = morgan_fingerprint(item["SMILES"])

    def generate_morgan_fingerprints(self) -> None:
        """Generates Morgan fingerprints for all data."""

        self._generate_morgan_fingerprints(self.train)
        self._generate_morgan_fingerprints(self.test)
        self._generate_morgan_fingerprints(self.val)

    def _write_lit_llama(self, path: str, data: DType) -> None:
        """Writes a data file in the format compatible with Lit-LLaMA."""

        data = [{"input": row["SMILES"], "output": row["description"]} for row in data]
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    def write_lit_llama(self) -> None:
        """Writes all data files in the format compatible with Lit-LLaMA."""

        dir = pathlib.Path("data")
        dir.mkdir(exist_ok=True)

        self._write_lit_llama(dir / "train.json", self.val)
        self._write_lit_llama(dir / "test.json", self.test)
        self._write_lit_llama(dir / "val.json", self.train)


def tanimoto_table(train: ChEBI20.DType, test: ChEBI20.DType, k: int) -> dict[str, list[int | str]]:
    """Generates a Tanimoto similarity table."""

    data = defaultdict(list)
    for item1 in tqdm.tqdm(test, desc="Generating Tanimoto similarity table"):
        for item2 in train:
            m1, s1 = item1["MORGAN"], item1["SMILES"]
            m2, s2, txt2 = item2["MORGAN"], item2["SMILES"], item2["description"]
            sim = DataStructs.TanimotoSimilarity(m1, m2)
            heapq.heappush(data[s1], (-sim, s2, txt2))

    neighbors = {
        key: [{"SIM": -sim, "SMILES": smiles, "CAP": cap} for (sim, smiles, cap) in val[:k]]
        for key, val in data.items()
    }

    return neighbors
