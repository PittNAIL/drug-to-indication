#!/usr/bin/env python
import json

import util


# Number most similar SMILES codes to retain
TOP_K: int = 10


def main() -> None:
    """Validates ChEBI-20 dataset, converts to Lit-LLaMA ️format, and writes resulting JSON files."""

    args = util.parse_args()

    cheb20 = util.ChEBI20(args.train, args.test, args.val)
    cheb20.generate_morgan_fingerprints()

    tt = util.tanimoto_table(cheb20.train + cheb20.val, cheb20.test, TOP_K)
    with open("data.json", "w", encoding="utf-8") as file:
        json.dump(tt, file, indent=2)


if __name__ == "__main__":
    main()
