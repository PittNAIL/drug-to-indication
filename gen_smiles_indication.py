#!/usr/bin/env python
import argparse
import json
import logging
import pathlib

import tqdm

import xml.etree.ElementTree as ET

import util


def parse_args() -> argparse.Namespace:
    """Parses the command line arguments."""

    parser = argparse.ArgumentParser("Drug List Generator")

    parser.add_argument("--drugbank", type=str, help="path to the DrugBank database", required=True)
    parser.add_argument("--out_json", type=str, help="path to write the JSON file", required=True)

    return parser.parse_args()


def main() -> None:
    """Extracts type, id, name, description, smiles, and indication from the DrugBank database."""

    args = parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f"{pathlib.Path(__file__).stem}.log", level=logging.INFO)

    logger.info("Loading and parsing DrugBank database...")
    tree = ET.parse(args.drugbank)
    root = tree.getroot()
    logger.info("Loading and parsing finished!")

    ns = "{http://www.drugbank.ca}"

    drugs = []
    for drug in tqdm.tqdm(root, desc="Walking through DrugBank database"):
        drugs.append(
            {
                "type": drug.get("type"),
                "drugbank_id": drug.findtext(f"{ns}drugbank-id[@primary='true']"),
                "name": drug.findtext(f"{ns}name"),
                "description": drug.findtext(f"{ns}description"),
                "smiles": drug.findtext(
                    f"{ns}calculated-properties/{ns}property[{ns}kind='SMILES']/{ns}value"
                ),
                "indication": drug.findtext(f"{ns}indication"),
            }
        )

    logger.info(f"{len(drugs)} entries before filtering")
    filtered_drugs = []
    for drug in tqdm.tqdm(drugs, desc="Filtering and canonicalizing DrugBank database..."):
        if not (drug["smiles"] and drug["indication"]):
            continue
        try:
            drug["canonical_smiles"] = util.canonicalize_smiles(drug["smiles"])
            filtered_drugs.append(drug)
        except Exception:
            pass
    logger.info(f"{len(drugs)} entries after filtering")

    logger.info(f"Writing to a {args.out_json}...")
    with open(args.out_json, "w", encoding="utf-8") as file:
        json.dump(filtered_drugs, file, ensure_ascii=False, indent=2)
    logger.info("Writing finished!")


if __name__ == "__main__":
    main()
