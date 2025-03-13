import os
import time
import click
import requests
import pandas as pd
from d2draftnet.config import MATCH_DATA_PATH  # Import config for the match data path

def load_existing_data():
    """
    Load existing match data from the Parquet file.
    """
    if os.path.exists(MATCH_DATA_PATH):
        return pd.read_parquet(MATCH_DATA_PATH)
    return pd.DataFrame(columns=["match_id", "radiant_draft", "dire_draft", "winner"])

def save_data(df):
    """
    Save the DataFrame to the Parquet file.
    """
    df.to_parquet(MATCH_DATA_PATH, index=False)

def get_open_dota_hero_map():
    """
    Retrieve hero mapping from the OpenDota API.
    Returns a dictionary mapping hero_id to hero localized_name.
    """
    url = "https://api.opendota.com/api/heroes"
    response = requests.get(url)
    if response.status_code == 200:
        heroes = response.json()
        return {hero["id"]: hero["localized_name"] for hero in heroes}
    else:
        click.echo("Error fetching hero map from OpenDota API")
        return {}

def get_match_details(match_id):
    """
    Retrieve match details for a given match_id.
    """
    url = f"https://api.opendota.com/api/matches/{match_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def extract_draft(match_data, hero_map):
    """
    Extract hero picks from match_data, ensuring exactly five picks per team.
    Convert hero IDs to names using hero_map.
    Returns (radiant_picks, dire_picks, winner) or (None, None, None) if invalid.
    """
    picks = match_data.get("picks_bans", [])
    radiant_ids = [entry["hero_id"] for entry in picks if entry.get("is_pick") and entry.get("team") == 0]
    dire_ids = [entry["hero_id"] for entry in picks if entry.get("is_pick") and entry.get("team") == 1]
    
    if len(radiant_ids) != 5 or len(dire_ids) != 5:
        return None, None, None
    
    radiant_picks = [hero_map.get(hero_id, str(hero_id)) for hero_id in radiant_ids]
    dire_picks = [hero_map.get(hero_id, str(hero_id)) for hero_id in dire_ids]
    winner = "Radiant" if match_data.get("radiant_win") else "Dire"
    
    return radiant_picks, dire_picks, winner


@click.command()
@click.option("--rows", default=10, help="Number of rows to display from the Parquet file.")
def view_parquet(rows):
    """
    Load and display data from the match data Parquet file.
    """
    try:
        df = pd.read_parquet(MATCH_DATA_PATH)
    except Exception as e:
        click.echo(f"Error reading Parquet file: {e}")
        return

    click.echo(df.head(rows))

@click.command()
@click.option("--limit", default=5, help="Number of matches to process.")
def main(limit):
    # Load existing match data and set of known match IDs.
    df_existing = load_existing_data()
    existing_match_ids = set(df_existing["match_id"]) if not df_existing.empty else set()
    
    # Get hero mapping from OpenDota API.
    hero_map = get_open_dota_hero_map()
    
    pro_matches_url = "https://api.opendota.com/api/proMatches"
    response = requests.get(pro_matches_url)
    if response.status_code != 200:
        click.echo("Error fetching pro matches")
        return
    
    matches = response.json()[:limit]
    records = []
    
    with click.progressbar(matches, label="Processing matches") as bar:
        for match in bar:
            match_id = match.get("match_id")
            if match_id in existing_match_ids:
                continue
            match_data = get_match_details(match_id)
            if not match_data or "picks_bans" not in match_data:
                continue
            radiant_picks, dire_picks, winner = extract_draft(match_data, hero_map)
            if radiant_picks is None:
                continue
            records.append({
                "match_id": match_id,
                "radiant_draft": radiant_picks,
                "dire_draft": dire_picks,
                "winner": winner
            })
            time.sleep(1)  # Throttle requests
    
    if records:
        df_new = pd.DataFrame(records)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["match_id"])
        save_data(df_combined)
        click.echo(f"Added {len(df_new)} new records. Total records: {len(df_combined)}")
    else:
        click.echo("No new records added.")

if __name__ == "__main__":
    #main()
    view_parquet()
