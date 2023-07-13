#!/usr/bin/env python
from datasets import Dataset

from transformers import AutoConfig, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

from validate import read_and_validate


CONTEXT_LENGTH: int = 128


def main() -> None:
    """Trains GPT-2 model on SMILES strings from PubChem."""

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(examples, context_length: int) -> object:
        """Helper function for tokenizing datasets."""

        return tokenizer(
            examples["train"],
            max_length=context_length,
            padding="max_length",
            truncation=True,
        )

    smiles_dataset = Dataset.from_dict({"train": read_and_validate("data/CID-SMILES")})
    tokenized_datasets = smiles_dataset.map(
        lambda data: tokenize(data, CONTEXT_LENGTH),
        batched=True,
        remove_columns=["train"],
    )

    config = AutoConfig.from_pretrained(
        "gpt2",
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        n_ctx=CONTEXT_LENGTH,
        vocab_size=len(tokenizer),
    )

    model = GPT2LMHeadModel(config)
    num_params = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {num_params / 1000**2:.1f}M parameters")

    args = TrainingArguments(
        fp16=True,
        learning_rate=2e-5,
        logging_steps=5_000,
        num_train_epochs=1_000,
        output_dir="smiles-gpt2",
        per_device_train_batch_size=32,
        save_steps=5_000,
        weight_decay=0.1,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        args=args,
        data_collator=data_collator,
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_datasets,
    )

    trainer.train()


if __name__ == "__main__":
    main()
