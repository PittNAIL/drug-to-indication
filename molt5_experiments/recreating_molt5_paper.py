import argparse
import json
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, set_seed
import tqdm
from sklearn.model_selection import train_test_split

models = ["small", "base", "large"]
SEED: int = 1_337
tasks = ["caption2smiles", "smiles2caption"]

set_seed(SEED)


def main() -> None:
    drugs_test = pd.read_csv(
        "/home/joh195/experiments/MolT5/ChEBI-20_data/test.txt", delimiter="\t"
    )

    for model in models:
        model_name = f"molt5-{model}"

        for task in tasks:
            print(f"Processing {model} on {task}.")
            tokenizer = T5Tokenizer.from_pretrained(
                f"laituan245/{model_name}-{task}", model_max_length=512
            )
            molt_model = T5ForConditionalGeneration.from_pretrained(
                f"laituan245/{model_name}-{task}"
            ).to("cuda")
            if task == "smiles2caption":
                model_data = []
                for _, drug in tqdm.tqdm(
                    drugs_test.iterrows(), total=len(drugs_test), desc="Processing drugs..."
                ):
                    df_cols = ["SMILES", "description", "output"]
                    input_text = drug["SMILES"]
                    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
                    outputs = molt_model.generate(input_ids, num_beams=5, max_length=512).to("cuda")

                    drug["output"] = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    model_data.append([drug[col] for col in df_cols])
                df_model = pd.DataFrame(model_data, columns=["SMILES", "ground truth", "output"])
                df_model.to_csv(f"recreated_{model_name}_{task}.txt", sep="\t", index=False)

                # Append data for a specific drug
            elif task == "caption2smiles":
                model_data = []
                for _, drug in tqdm.tqdm(
                    drugs_test.iterrows(), total=len(drugs_test), desc="Processing drugs..."
                ):
                    df_cols = ["description", "SMILES", "output"]
                    input_text = drug["description"]
                    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
                    outputs = molt_model.generate(input_ids, num_beams=5, max_length=512).to("cuda")
                    drug["model"] = model_name
                    drug["output"] = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    model_data.append([drug[col] for col in df_cols])
                df_model = pd.DataFrame(
                    model_data, columns=["description", "ground truth", "output"]
                )
                df_model.to_csv(f"recreated_{model_name}_{task}.txt", sep="\t", index=False)


if __name__ == "__main__":
    main()
