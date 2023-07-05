import argparse
import csv
import json
import pathlib


def parse_args() -> argparse.Namespace:
    """Parses the command line arguments."""

    parser = argparse.ArgumentParser("Automatic captioning of molecules")

    parser.add_argument("--train", type=str, help="path to training dataset", required=True)
    parser.add_argument("--test", type=str, help="path to testing dataset", required=True)
    parser.add_argument("--val", type=str, help="path to validation dataset", required=True)

    return parser.parse_args()


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
