#Custom Model Creation

We used the following script for creating the custom Tokenizer MolT5 model:
```console
python run_t5_mlm_flax.py --output_dir="./molt5-small" --model_type="t5" --config_name="./molt5-small" --tokenizer_name="./molt5-small" --train_file="zinc_smiles_train.txt" --validation_file="zinc_smiles_val.txt" --max_seq_length="512" --per_device_train_batch_size="4" --per_device_eval_batch_size="4" --adafactor --learning_rate="0.005" --weight_decay="0.001" --warmup_steps="2000" --overwrite_output_dir --logging_steps="1000" --save_steps="40000" --eval_steps="10000"
```
Note: this will require downloading run_t5_mlm_flax.py from the transformers repo, as well as
obtaining zinc_smiles_train.txt and zinc_smiles_test.txt from the MolT5 repo.
