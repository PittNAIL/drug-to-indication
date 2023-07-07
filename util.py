import argparse
import csv
import heapq
import json
import pathlib

import tqdm

from collections import defaultdict

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdMolDescriptors

from sentence_transformers import SentenceTransformer, util


RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    """Parses the command line arguments."""

    parser = argparse.ArgumentParser("Automatic captioning of molecules")

    parser.add_argument("--train", type=str, help="path to training dataset", required=True)
    parser.add_argument("--test", type=str, help="path to testing dataset", required=True)
    parser.add_argument("--val", type=str, help="path to validation dataset", required=True)
    parser.add_argument("--top_k", type=int, help="most similar SMILES to retain", required=True)
    parser.add_argument("--model", type=str, help="transformer model to use", required=True)

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

        self.train_path, self.test_path, self.val_path = train_path, test_path, val_path

        self.train = self._read(self.train_path)
        assert len(self.train) == 26_407

        self.test = self._read(self.test_path)
        assert len(self.test) == 3_300

        self.val = self._read(self.val_path)
        assert len(self.val) == 3_301

    def _read(self, path: str) -> DType:
        """Reads ChEBI-20 data files."""

        with open(path) as file:
            data = list(csv.DictReader(file, delimiter="\t"))

        return data

    def _generate_morgan_fingerprints(self, data: DType, label: str) -> None:
        """Generates Morgan fingerprints."""

        for item in tqdm.tqdm(data, desc=f"Generating Morgan fingerprints for {label}"):
            item["MORGAN"] = morgan_fingerprint(item["SMILES"])

    def generate_morgan_fingerprints(self) -> None:
        """Generates Morgan fingerprints for all data."""

        self._generate_morgan_fingerprints(self.train, self.train_path)
        self._generate_morgan_fingerprints(self.test, self.test_path)
        self._generate_morgan_fingerprints(self.val, self.val_path)

    def _write_llama(self, path: str, data: DType) -> None:
        """Writes a data file in the format compatible with LLaMA."""

        data = [
            {
                "input": row["description"],
                "output": row["SMILES"],
                "instruction": "Generate a SMILES string for a following description: ",
            }
            for row in data
        ]
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    def write_llama(self) -> None:
        """Writes all data files in the format compatible with LLaMA."""

        dir = pathlib.Path("data")
        dir.mkdir(exist_ok=True)

        self._write_llama(dir / "train.json", self.val)
        self._write_llama(dir / "test.json", self.test)
        self._write_llama(dir / "val.json", self.train)


TanimotoRankingTable = dict[str, str | list[dict[float | str]]]


def tanimoto_ranking(
    train: ChEBI20.DType, test: ChEBI20.DType, k: int, save: bool = False
) -> TanimotoRankingTable:
    """Computes rankings with k most semantically similar entries using Tanimoto similarity."""

    data = defaultdict(list)
    for item1 in tqdm.tqdm(test, desc="Computing Tanimoto similarity rankings"):
        for item2 in train:
            m1, s1, txt1 = item1["MORGAN"], item1["SMILES"], item1["description"]
            m2, s2, txt2 = item2["MORGAN"], item2["SMILES"], item2["description"]
            sim = DataStructs.TanimotoSimilarity(m1, m2)
            heapq.heappush(data[(s1, txt1)], (-sim, s2, txt2))

    neighbors = {
        s1: {
            "CAP": txt1,
            "INFO": [{"SIM": -sim, "SMILES": s2, "CAP": txt2} for (sim, s2, txt2) in val[:k]],
        }
        for (s1, txt1), val in data.items()
    }

    # NOTE: Writes JSON data as a backup
    if save:
        dir = pathlib.Path("data")
        dir.mkdir(exist_ok=True)
        with open(dir / "tanimoto_rankings.json", "w", encoding="utf-8") as file:
            json.dump(neighbors, file, indent=2)

    return neighbors


def sentence_transformer(model: str) -> SentenceTransformer:
    """Gets the codename by the model name."""

    match model:
        case "bert":
            codename = "bert-base-cased"
        case "biobert":
            codename = "dmis-lab/biobert-base-cased-v1.2"
        case "bioclinicalbert":
            codename = "emilyalsentzer/Bio_ClinicalBERT"
        case _:
            raise ValueError(f"Unknown model: {model}")

    return SentenceTransformer(codename)


CosineRankingTable = dict[str, str | list[dict[float | str]]]


def cosine_ranking(
    model: str, tt: dict[str, list[int | str]], k: int, save: bool = False
) -> CosineRankingTable:
    """Computes rankings with k most semantically similar entries using cosine similarity."""

    model = sentence_transformer(model)

    data = defaultdict(list)
    for s1, info in tqdm.tqdm(tt.items(), desc="Computing cosine similarity rankings"):
        txt1 = info["CAP"]
        emb1 = model.encode(txt1, convert_to_tensor=True)
        for item in info["INFO"]:
            s2, txt2 = item["SMILES"], item["CAP"]
            emb2 = model.encode(txt2, convert_to_tensor=True)
            heapq.heappush(data[(s1, txt1)], (-util.cos_sim(emb1, emb2).item(), s2, txt2))

    neighbors = {
        s1: {
            "CAP": txt1,
            "INFO": [{"SIM": -sim, "SMILES": s2, "CAP": txt2} for (sim, s2, txt2) in val[:k]],
        }
        for (s1, txt1), val in data.items()
    }

    # NOTE: Writes JSON data as a backup
    if save:
        dir = pathlib.Path("data")
        dir.mkdir(exist_ok=True)
        with open(dir / "cosine_similarity_rankings.json", "w", encoding="utf-8") as file:
            json.dump(neighbors, file, indent=2)

    return neighbors
