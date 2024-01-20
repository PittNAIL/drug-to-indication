import pickle
import tqdm

import pandas as pd
import pubchempy as pcp

df = pd.read_csv("/home/joh195/experiments/chembl_indications_smiles.csv")

smiles = df["canonical_smiles"]

cids = []

for index, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    smile = row["canonical_smiles"]
    try:
        cid = pcp.get_cids(smile, "smiles")[0]
        cids.append(cid)
    except Exception as e:
        print(f"'smiles' retrieval failed at index {index}.")
        cids.append("ERROR")
        df.drop(index)

data_dict = dict(zip(cids, smiles))

with open("chembl_cids.pkl", "wb") as handle:
    pickle.dump(data_dict, handle)

with open("chembl_cids.pkl", "rb") as f:
    cids_to_smiles = pickle.load(f)

df = df[df["canonical_smiles"].isin(cids_to_smiles.values())]

df.to_csv("chembl_data.csv")
