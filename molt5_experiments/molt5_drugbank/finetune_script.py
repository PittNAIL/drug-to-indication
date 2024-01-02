import evaluate
import json
import torch
import tqdm

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration
from transformers import set_seed, AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM


SEED: int = 1_337

set_seed(SEED)

models = ["small", "base", "large"]
tasks = ["smiles2caption", "caption2smiles"]


class MyDataset(Dataset):
    def __init__(self, data, tokenizer, task, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.task = task
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = (
            item["indication"] if self.task == "caption2smiles" else item["canonical_smiles"]
        )
        target_text = (
            item["canonical_smiles"] if self.task == "caption2smiles" else item["indication"]
        )

        # Tokenize and pad/truncate the input and target sequences
        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer.encode_plus(
            target_text,
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
    f = open("drugbank.json", encoding="UTF-8")
    drugs = json.load(f)
    drugs_train, drugs_test = train_test_split(drugs, test_size=0.2, random_state=SEED)

    for task in tasks:
        for model in models:
            model_name = f"{model}"
            # tokenizer = T5Tokenizer.from_pretrained(f"{model}", model_max_length=512)
            tokenizer = AutoTokenizer.from_pretrained(
                f"laituan245/molt5-{model}", model_max_length=512
            )
            molt_model = T5ForConditionalGeneration.from_pretrained(f"laituan245/molt5-{model}").to(
                "cuda"
            )
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
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                gradient_accumulation_steps=4,
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

            if task == "caption2smiles":
                df_cols = ["indication", "canonical_smiles", "output"]
                fine_tuned_model = T5ForConditionalGeneration.from_pretrained(
                    f"{output_directory}/checkpoint-900"
                ).to("cuda")
                model_data = []
                for drug in tqdm.tqdm(drugs_test, desc="Evaluating drugs..."):
                    input_text = drug["indication"]
                    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
                    outputs = fine_tuned_model.generate(input_ids, num_beams=5, max_length=512).to(
                        "cuda"
                    )
                    drug["output"] = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    model_data.append([drug[col] for col in df_cols])

                df_model = pd.DataFrame(
                    model_data, columns=["description", "ground truth", "output"]
                )
                df_model.to_csv(f"fine_tuned_{model_name}.txt", sep="\t", index=False)

            if task == "smiles2caption":
                df_cols = ["canonical_smiles", "indication", "output"]
                fine_tuned_model = T5ForConditionalGeneration.from_pretrained(
                    f"{output_directory}/checkpoint-900"
                ).to("cuda")
                model_data = []
                for drug in tqdm.tqdm(drugs_test, desc="Evaluating drugs..."):
                    input_text = drug["canonical_smiles"]
                    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
                    outputs = fine_tuned_model.generate(input_ids, num_beams=5, max_length=512).to(
                        "cuda"
                    )
                    drug["output"] = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    model_data.append([drug[col] for col in df_cols])

                df_model = pd.DataFrame(model_data, columns=["SMILES", "ground truth", "output"])
                df_model.to_csv(f"fine_tuned_{model_name}.txt", sep="\t", index=False)


if __name__ == "__main__":
    main()
