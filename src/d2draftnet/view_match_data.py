from .config import MATCH_DATA_PATH
import pandas as pd

def check_repeats(df: pd.DataFrame) -> int:
    """
    Check for repeated match IDs in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing match data.
        
    Returns:
        int: Number of repeated match IDs.
    """
    return df['id'].duplicated().sum()

def main():
    try:
        df = pd.read_parquet(MATCH_DATA_PATH)
    except Exception as e:
        print(f"Error reading {MATCH_DATA_PATH}: {e}")
        return

    print(f"Loaded {len(df)} matches from {MATCH_DATA_PATH}.")
    print(check_repeats(df), "repeated matches found.")
    print("Sample match data:\n")
    sample_match = df.iloc[1]
    for key, value in sample_match.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
