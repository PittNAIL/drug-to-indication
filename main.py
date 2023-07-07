#!/usr/bin/env python
import util


def main() -> None:
    """Implements inference and evaluation pipeline."""

    # STEP 0: Parse command line arguments
    args = util.parse_args()

    # STEP 1: Load the data, writes a dataset for LLaMA, and generates Morgan fingerprints
    cheb20 = util.ChEBI20(args.train, args.test, args.val)
    cheb20.write_llama()
    cheb20.generate_morgan_fingerprints()

    # STEP 2: Generate Tanimoto similarity table and retain molecules with TOP k most similar SMILES
    tt = util.tanimoto_ranking(cheb20.train + cheb20.val, cheb20.test, args.top_k, save=True)

    # STEP 3: Generate embeddings and use semantic similarity to retain TOP k / 2 SMILES
    st = util.cosine_ranking(args.model, tt, args.top_k // 2, save=True)

    print(f"{len(st)=}")


if __name__ == "__main__":
    main()
