import psycopg2
import argparse
import pandas as pd
from io import StringIO


def connect_to_postgres(dbname, user, password, host):
    try:
        connection = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)

        cursor = connection.cursor()

        query = """                                                       
  SELECT canonical_smiles, indication_name
  FROM (
    SELECT
      cs.canonical_smiles,
      di.efo_term as indication_name
    FROM
      chembl_id_lookup ch
      JOIN molecule_dictionary md ON ch.chembl_id = md.chembl_id
      JOIN drug_indication di ON md.molregno = di.molregno
      JOIN compound_structures cs ON md.molregno = cs.molregno
    WHERE
      ch.entity_type = 'COMPOUND' AND
      ch.status = 'ACTIVE' AND
      md.pref_name IS NOT NULL
  ) as subquery;                                                              
"""
        cursor.execute(query)

        records = cursor.fetchall()

        # Save the records to a CSV file
        df = pd.DataFrame(records, columns=[desc[0] for desc in cursor.description])
        df.to_csv("chembl_data.csv", index=False)

        print("CSV file saved successfully.")

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
