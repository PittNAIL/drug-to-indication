import evaluate
import json
import os
import torch
import tqdm

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import set_seed, T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments

SEED: int = 1_337

set_seed(SEED)

models = ["molt5-small", "molt5-base", "molt5-large"]
tasks = ["indication2smiles", "smiles2indication"]
#methods = ["subset", "full", "finetuned"]
methods = ["finetuned"]
class PrepDataset(Dataset):
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
            item["indication"] if self.task == "indication2smiles" else item["canonical_smiles"]
        )
        target_text = (
            item["canonical_smiles"] if self.task == "smiles2indication" else item["indication"]
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

def eval_drugs(task, model, data, method, tokenizer, finetuned_train=None):
    print(f"Evaluating {model} on {task}, {method} experiment")
    if method == "finetuned":
        model = T5ForConditionalGeneration.from_pretrained(f"laituan245/molt5-{model}").to("cuda")
        tokenizer = T5Tokenizer(f"laituan245/molt5-{model}")
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
        output_directory = f"fine_tuned_{model}_{task}"
        training_args = TrainingArguments(
        output_dir = output_directory,
        overwrite_output_dir = False,
        do_train = True,
        do_eval = True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-5,
        num_train_epochs=3,
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        )
        train_dataset = PrepDataset(finetuned_train, tokenizer, task)
        test_dataset = PrepDataset(data, tokenizer, task)

        trainer = Trainer(
                model = model,
                args = training_args,
                train_dataset = train_dataset,
                test_dataset = test_dataset,
                )
        trainer.train()
        if task == "indication2smiles":
            ft_model = T5ForConditionalGeneration(f"{output_directory}/checkpoint-900").to("cuda")
            model_data = {"description": [], "ground truth": [], "output": []}
            for drug in tqdm.tqdm(data, desc = "Evaluating drugs..."):
                input_text = drug["indication"]
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
                outputs = ft_model.generate(input_ids, num_beams=5, max_length=512).to("cuda")
                output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                model_data["description"].append(drug["indication"])
                model_data["ground truth"].append(drug["canonical_smiles"])
                model_data["output"].append(output)
            df_model = pd.DataFrame(model_data)
            df_model.to_csv(f"drugbank_{method}_molt5-{model}_{task}.txt", sep="\t", index=False)
        if task == "smiles2indication":
            ft_model = T5ForConditionalGeneration(f"{output_directory}/checkpoint-900").to("cuda")
            model_data = {"SMILES": [], "ground truth": [], "output": []}
            for drug in tqdm.tqdm(data, desc = "Evaluating drugs..."):
                input_text = drug["canonical_smiles"]
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
                outputs = ft_model.generate(input_ids, num_beams=5, max_length=512).to("cuda")
                output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                model_data["SMILES"].append(drug["canonical_smiles"])
                model_data["ground truth"].append(drug["indication"])
                model_data["output"].append(output)
            df_model = pd.DataFrame(model_data)
                           
