import evaluate
import json
import torch
import tqdm
import os

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration
from transformers import set_seed, AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM


SEED: int = 1_337

set_seed(SEED)

models = ["laituan245/molt5-small"]
tasks = ["smiles2indication", "indication2smiles"]

class MyDataset(Dataset):
    def __init__(self, data, tokenizer, task, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.task = task
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        if self.task == "indication2smiles":
            input_text = item["indication_name"]
            target_text = item["canonical_smiles"]
        else:
            input_text = item["canonical_smiles"]
            target_text = item["indication_name"]

        # Tokenize and pad/truncate the input and target sequences
        inputs = self.tokenizer.encode_plus(
            text=str(input_text),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer.encode_plus(
            text=str(target_text),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": targets["input_ids"].flatten(),
        }


def main() -> None:
    drugs = pd.read_csv("imputed_chembl.csv")
    drugs_train, drugs_test = train_test_split(drugs, test_size=0.2, random_state=SEED)

    for task in tasks:
        for model in models:
            if model == "laituan245/molt5-small":
                model_name = "molt5-small"
            else:
                model_name = model
            tokenizer = AutoTokenizer.from_pretrained(f"{model}", model_max_length=512)
            molt_model = AutoModelForSeq2SeqLM.from_pretrained(f"{model}").to("cuda")
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                molt_model.resize_token_embeddings(len(tokenizer))
            output_directory = f"fine_tuned_{model_name}_{task}"
            training_args = TrainingArguments(
                output_dir=output_directory,
                overwrite_output_dir=False,
                do_train=True,
                do_eval=True,
                evaluation_strategy="epoch",
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                gradient_accumulation_steps=16,
                learning_rate=5e-5,
                num_train_epochs=3,
                logging_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
            )
            train_dataset = MyDataset(drugs_train, tokenizer, task)
            test_dataset = MyDataset(drugs_test, tokenizer, task)

            trainer = Trainer(
                model=molt_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
            )

            trainer.train()

            if task == "indication2smiles":
                df_cols = ["indication_name", "target_name"]
                model_dir = f"{output_directory}/checkpoint-36"
                ft_model = AutoModelForSeq2SeqLM.from_pretrained(f"{model_dir}").to("cuda")
                model_data = {"indication": [], "ground truth": [], "output": []}
                for _, drug in tqdm.tqdm(
                    drugs_test.iterrows(), total=len(drugs_test), desc="Evaluating drugs..."
                ):
                    input_text = drug["indication_name"]
                    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
                    outputs = ft_model.generate(input_ids, num_beams=5, max_length=512).to("cuda")
                    output_final = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    model_data["indication"].append(drug["indication_name"])
                    model_data["ground truth"].append(drug["canonical_smiles"])
                    model_data["output"].append(output_final)
                df_model = pd.DataFrame(model_data, columns = ["description", "ground truth",
                                                               "output"])
                df_model.to_csv(f"chembl_fine_tuned_{model_name}_{task}.txt", sep="\t", index=False)

            if task == "smiles2indication":
                df_cols = ["target_name", "indication_name"]
                model_dir = f"{output_directory}/checkpoint-36"
                ft_model = AutoModelForSeq2SeqLM.from_pretrained(f"{model_dir}").to("cuda")
                model_data = {"canonical_smiles": [], "ground truth": [], "output": []}
                for _, drug in tqdm.tqdm(
                    drugs_test.iterrows(), total=len(drugs_test), desc="Evaluating drugs..."
                ):
                    input_text = drug["canonical_smiles"]
                    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
                    outputs = ft_model.generate(input_ids, num_beams=5, max_length=512).to("cuda")
                    output_final = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    model_data["canonical_smiles"].append(drug["canonical_smiles"])
                    model_data["ground truth"].append(drug["indication_name"])
                    model_data["output"].append(output_final)
                df_model = pd.DataFrame(model_data, columns = ["SMILES", "ground truth", "output"])
                df_model.to_csv(f"chembl_fine_tuned_{model_name}_{task}.txt", sep="\t", index=False)


if __name__ == "__main__":
    main()
