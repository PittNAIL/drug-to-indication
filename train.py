import util

from datasets import Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

DATASET_SIZE: int = 10_000
N_POSITIONS: int = 512
N_LAYER = 6  # Number of transformer layers
N_HEAD = 8  # Number of multi-head attention heads
N_EMBD = 256  # Embedding size


def read(path: str, size: int) -> list[str]:
    """Reads SMILES strings from PubChem."""

    data = []
    with open(path) as file, tqdm(total=size, desc=f"Reading {path}...") as pbar:
        while (line := file.readline()) and len(data) < size:
            smiles = line.split()[1]
            smiles = util.canonicalize_smiles(smiles)
            bitstr = " ".join(list(util.maccs_fingerprint(smiles).ToBitString()))
            prompt = f"{bitstr}\n{smiles}"
            data.append(prompt)
            pbar.update(1)
    assert len(data) == size

    return data


dataset = Dataset.from_dict({"prompt": read("CID-SMILES", DATASET_SIZE)})

tokenizer = AutoTokenizer.from_pretrained("gpt2")


def tokenize(elem: dict[str, str]) -> dict[str, list[int]]:
    out = tokenizer(
        elem["prompt"],
        truncation=True,
        max_length=N_POSITIONS,
        return_overflowing_tokens=True,
        return_length=True,
    )

    return {"input_ids": out["input_ids"]}


tok_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig

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

args = TrainingArguments(
    output_dir="maccs_models",
    per_device_train_batch_size=32,
    logging_steps=100,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tok_dataset["input_ids"],
)

trainer.train()
