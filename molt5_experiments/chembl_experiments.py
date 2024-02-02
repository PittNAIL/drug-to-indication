import tqdm

import pandas as pd

from sklearn.model_selection import train_test_split    
from torch.utils.data import Dataset
from transformers import set_seed, T5ForConditionalGeneration, T5Tokenizer
from transformers import Trainer, TrainingArguments

SEED: int = 1_337

set_seed(SEED)

models = ["molt5-small", "molt5-base", "molt5-large"]
tasks = ["indication2smiles", "smiles2indication"]
methods = ["subset", "full", "finetuned"]

class PrepDataset(Dataset):
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

def eval_drugs(task, model, data, method):
    if method == "finetuned":
        task_model = T5ForConditionalGeneration.from_pretrained(f"{model}").to("cuda")
    else:
        if task == "indication2smiles":
            task_model = T5ForConditionalGeneration.from_pretrained(
                f"laituan245/{model}-caption2smiles").to("cuda")
            tokenizer = T5Tokenizer.from_pretrained(f"laituan245/{model}-caption2smiles")
        if task == "smiles2indication":    
            task_model = T5ForConditionalGeneration.from_pretrained(
                f"laituan245/{model}-smiles2caption"
                ).to("cuda")
            tokenizer = T5Tokenizer.from_pretrained(f"laituan245/{model}-smiles2caption")

    if task == "indication2smiles":
        model_data = {"description": [], "ground truth": [], "output": []}
        for _, drug in tqdm.tqdm(data.iterrows(), total=len(data), desc = "Generating SMILES from Indications..."):
            input_text = drug["indication_name"]
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
            outputs = task_model.generate(input_ids, num_beams=5, max_length=512).to("cuda")
            output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            model_data["description"].append(drug["indication_name"])
            model_data["ground truth"].append(drug["canonical_smiles"])
            model_data["output"].append(output)
        df_model = pd.DataFrame(model_data)
        df_model.to_csv(f"chembl_{method}_{model}_{task}.txt", sep="\t", index=False)
    if task == "smiles2indication":
        model_data = {"SMILES": [], "ground truth": [], "output": []}
        for _, drug in tqdm.tqdm(data.iterrows(), total=len(data), desc = "Generating Indications from SMILES..."):
            input_text = drug["canonical_smiles"]
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
            outputs = task_model.generate(input_ids, num_beams=5, max_length=512).to("cuda")
            output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            model_data["SMILES"].append(drug["canonical_smiles"])
            model_data["ground truth"].append(drug["indication_name"])
            model_data["output"].append(output)
        df_model = pd.DataFrame(model_data)
        df_model.to_csv(f"chembl_{method}_{model}_{task}.txt", sep="\t", index=False)

def main() -> None:
    drugs = pd.read_csv("imputed_chembl.csv")
    drugs_train, drugs_test = train_test_split(drugs, test_size = 0.2, random_state=SEED)
    for method in methods:
        if method == "finetuned":
            for task in tasks:
                for model in models:
                    molt_model = T5ForConditionalGeneration.from_pretrained(f"laituan245/{model}")
                    tokenizer = T5Tokenizer.from_pretrained(f"laituan245/{model}")
                    output_directory = f"fine_tuned_{model}_{task}"
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
                    train_dataset = PrepDataset(drugs_train, tokenizer, task)
                    test_dataset = PrepDataset(drugs_test, tokenizer, task)
                
                    trainer = Trainer(
                        model = molt_model,
                        args = training_args,
                        train_dataset = train_dataset,
                        eval_dataset = test_dataset,
                        )
                    print(f"Finetuning {model} on {task}.")
                    trainer.train()
                    ft_model = f"{output_directory}/checkpoint-36"
                    eval_drugs(task, ft_model, drugs_test, method)

        if method == "full":
            for model in models:
                for task in tasks:
                    print(f"Performing {task} on {method} dataset using {model}.")
                    eval_drugs(task, model, drugs, method)
        if method == "subset":
            for model in models:
                for task in tasks:
                    print(f"Performing {task} on {method} dataset using {model}.")
                    eval_drugs(task, model, drugs_test, method)
if __name__ == "__main__":
    main()
