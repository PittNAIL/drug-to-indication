import psycopg2
import argparse
import pandas as pd
from io import StringIO


def connect_to_postgres(dbname, user, password, host):
    try:
        # Establish a connection to the database
        connection = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)

        # Create a cursor object to interact with the database
        cursor = connection.cursor()

        # Read the query from the file
        query = """                                                       
  SELECT canonical_smiles, indication_name
  FROM (
    SELECT
      ch.chembl_id,
      ch.entity_id,
      md.pref_name,
      cs.canonical_smiles,
      di.efo_term as indication_name,
      ROW_NUMBER() OVER (PARTITION BY ch.chembl_id ORDER BY di.efo_term) as rnk
    FROM
      chembl_id_lookup ch
      JOIN molecule_dictionary md ON ch.chembl_id = md.chembl_id
      JOIN drug_indication di ON md.molregno = di.molregno
      JOIN compound_structures cs ON md.molregno = cs.molregno
    WHERE
      ch.entity_type = 'COMPOUND' AND
      ch.status = 'ACTIVE' AND
      md.pref_name IS NOT NULL
  ) ranked                                                       
  WHERE rnk <= 16;                                                              
"""
        # Execute the query
        cursor.execute(query)

        # Fetch the result
        records = cursor.fetchall()

        # Save the records to a CSV file
        df = pd.DataFrame(records, columns=[desc[0] for desc in cursor.description])
        grouped_df = (
            df.groupby(["canonical_smiles"])["indication_name"]
            .agg(lambda x: ", ".join(str(val) for val in x if not pd.isna(val)))
            .reset_index()
        )
        grouped_df["indication_name"] = (
            "Medical conditions treated by this drug: " + grouped_df["indication_name"]
        )
        grouped_df.to_csv("chembl_data.csv", index=False)

        print("CSV file saved successfully.")

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL or executing query", error)

    finally:
        # Close the connection and cursor
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
    # Set up command-line argument parsing
    main()

    # Parse the command-line arguments
