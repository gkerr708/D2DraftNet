from d2draftnet.config import MATCH_DATA_PATH
import pandas as pd

def main():
    try:
        df = pd.read_parquet(MATCH_DATA_PATH)
    except Exception as e:
        print(f"Error reading {MATCH_DATA_PATH}: {e}")
        return

    print(f"Loaded {len(df)} matches from {MATCH_DATA_PATH}.")
    #print("\nMatch Data (first 10 rows):")
    #print(df.head(10))
    sample_match = df.iloc[5]
    for key, value in sample_match.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
