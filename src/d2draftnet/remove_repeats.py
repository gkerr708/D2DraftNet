from pathlib import Path
import click as ck
import pandas as pd

from d2draftnet.config import MATCH_DATA_PATH

"""
Remove repeated entries from a JSON file.
"""

def remove_repeats(file_path: Path):
    """Removed match which do not a unique match ID."""

    # Load the data
    df = pd.read_parquet(file_path)

    # Get the total number of data points
    N_total = len(df)

    # Remove duplicates
    df.drop_duplicates(subset="id", keep="first", inplace=True)

    # Calculate the number of removed entries
    N_repeats = N_total - len(df)

    # Print the number of removed entries
    ck.secho(f"Removed {N_repeats} repeated entries from {N_total} total entries.", fg="green")

    # Save the changes if the user confirms
    if ck.confirm("Do you want to save the changes?"):
        df.to_parquet(file_path, index=False)
        ck.secho(f"Saved changes to {file_path}.", fg="green")
        ck.secho(f"Total entries: {len(df)}.", fg="green")

if __name__ == "__main__":
    remove_repeats(MATCH_DATA_PATH)





