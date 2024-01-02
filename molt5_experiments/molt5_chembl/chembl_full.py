import tqdm

import pandas as pd

from sklearn.model_selection import train_test_split
from transformers import set_seed, T5Tokenizer, T5ForConditionalGeneration


SEED: int = 1_337

set_seed(SEED)

models = ["molt5-small", "molt5-base", "molt5-large"]
tasks = ["indication2smiles", "smiles2indication"]


def main() -> None:
    drugs = pd.read_csv("imputed_chembl.csv")
    drugs_train, drugs_test = train_test_split(drugs, test_size=0.2, random_state=SEED)
    for task in tasks:
        for model in models:
            model_name = f"{model}"
            print(f"Evaluating {model} on {task}")
            if task == "indication2smiles":
                if (model == "molt5-small") & (task == "indication2smiles"):
                    print(f"skipping {model} on {task}")
                    continue
                if (model == "molt5-base") & (task == "indication2smiles"):
                    print(f"Skipping {model} on {task}")
                    continue
                model_state = f"laituan245/{model}-caption2smiles"
                molt_model = T5ForConditionalGeneration.from_pretrained(model_state).to("cuda")
                tokenizer = T5Tokenizer.from_pretrained(model_state)
                model_data = {"description": [], "ground truth": [], "output": []}
                for _, drug in tqdm.tqdm(
                    drugs.iterrows(), total=len(drugs), desc="Evaluating drugs..."
                ):
                    input_text = drug["indication_name"]
                    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
                    outputs = molt_model.generate(input_ids, num_beams=5, max_length=512).to("cuda")
                    output_final = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    model_data["description"].append(drug["indication_name"])
                    model_data["ground truth"].append(drug["canonical_smiles"])
                    model_data["output"].append(output_final)
                df_model = pd.DataFrame(model_data)
                df_model.to_csv(f"chembl_full_{model_name}_{task}.txt", sep="\t", index=False)
            if task == "smiles2indication":
                model_data = {"SMILES": [], "ground truth": [], "output": []}
                model_state = f"laituan245/{model}-smiles2caption"
                molt_model = T5ForConditionalGeneration.from_pretrained(model_state).to("cuda")
                tokenizer = T5Tokenizer.from_pretrained(model_state)
                for _, drug in tqdm.tqdm(
                    drugs.iterrows(), total=len(drugs_test), desc="Evaluating drugs..."
                ):
                    input_text = drug["canonical_smiles"]
                    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
                    outputs = molt_model.generate(input_ids, num_beams=5, max_length=512).to("cuda")
                    output_final = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    model_data["SMILES"].append(drug["canonical_smiles"])
                    model_data["ground truth"].append(drug["indication_name"])
                    model_data["output"].append(output_final)
                df_model = pd.DataFrame(model_data)
                df_model.to_csv(f"chembl_full_{model_name}_{task}.txt", sep="\t", index=False)


if __name__ == "__main__":
    main()
