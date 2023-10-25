import argparse
import json
import psycopg2

import pandas as pd
from io import StringIO


def connect_to_postgres(dbname, user, password, host):
    try:
        connection = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)

        cursor = connection.cursor()

        query = """                                                       
  SELECT
    md.pref_name,
    ms.synonyms,
    cs.canonical_smiles
  FROM
    molecule_synonyms ms 
    join compound_structures cs on ms.molregno=cs.molregno 
    join molecule_dictionary md on md.molregno = ms.molregno 
  WHERE
    md.pref_name IS NOT NULL AND 
    (ms.syn_type='OTHER' OR
     ms.syn_type='RESEARCH_CODE');                                                       
"""
        cursor.execute(query)

        records = cursor.fetchall()

        # Save the records to a CSV file
        df = pd.DataFrame(records, columns=[desc[0] for desc in cursor.description])
        
        grouped_df = df.groupby(['pref_name', 'canonical_smiles'])['synonyms'].agg(list).reset_index()
        
        chembl_json = {}
        
        for _, row in grouped_df.iterrows():
            drug = row['pref_name']
            smiles = row['canonical_smiles']
            synonyms = row['synonyms']
            drug_dict = {'smiles':smiles, 'synonyms':synonyms}
            chembl_json[drug] = drug_dict
        
        with open('chembl.json', 'w') as f:
            json.dump(chembl_json, f)
            
        print("JSON file saved successfully.")

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL or executing query", error)

    finally:
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")


def parse_args() -> argparse.Namespace():
    """Parse command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Connect to PostgreSQL database, execute a query, and save the result to CSV"
    )
    parser.add_argument("-d", "--dbname", help="Database name", required=True)
    parser.add_argument("-u", "--user", help="Database user", required=True)
    parser.add_argument("-p", "--password", help="Database password")
    parser.add_argument("-host", "--host", help="Database host", default="localhost")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.password is None:
        args.password = input("Enter database password: ")

    connect_to_postgres(args.dbname, args.user, args.password, args.host)


if __name__ == "__main__":
    main()
