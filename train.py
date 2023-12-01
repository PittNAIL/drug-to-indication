#!/usr/bin/env python
import torch

from datasets import load_dataset

from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import set_seed


SEED: int = 1_337
DATASET_SIZE: int = 10_000_000
BATCH_SIZE: int = 64
N_POSITIONS: int = 512
N_LAYER = 6
N_HEAD = 8
N_EMBD = 128

set_seed(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def main() -> None:
    """Trains custom GPT-2-based model on SMILES strings from PubChem."""

    pubchem_smiles = load_dataset(
        "csv",
        data_files=["pubchem_smiles.csv"],
        streaming=True,
    ).shuffle(SEED)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def tokenize(elem: dict[str, str]) -> dict[str, list[int]]:
        out = tokenizer(
            elem["prompt"],
            truncation=True,
            max_length=N_POSITIONS,
            return_overflowing_tokens=True,
        )

        return {"input_ids": out["input_ids"]}

    tok_dataset = pubchem_smiles.map(tokenize, batched=True, remove_columns=["prompt"])

    config = AutoConfig.from_pretrained(
        "gpt2",
        n_positions=N_POSITIONS,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = GPT2LMHeadModel(config)
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad_)
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # NOTE: we use `max_steps = grad_acc_steps * (num_epochs * dataset_size / batch_size)`
    args = TrainingArguments(
        output_dir="maccs_models",
        per_device_train_batch_size=BATCH_SIZE,
        logging_steps=1_000,
        gradient_accumulation_steps=4,
        max_steps=4 * int(DATASET_SIZE // BATCH_SIZE + 0.5),
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=1e-4,
        save_steps=1_000,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tok_dataset["train"],
    )

    trainer.train()
    trainer.save_model("maccs_models/final_model")


if __name__ == "__main__":
    main()
