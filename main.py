#!/usr/bin/env python
import util


def main() -> None:
    """Validates ChEBI-20 dataset, converts to Lit-LLaMA ️format, and writes resulting JSON files."""

    args = util.parse_args()

    cheb20 = util.ChEBI20(args.train, args.test, args.val)
    cheb20.write_lit_llama()


if __name__ == "__main__":
    main()
