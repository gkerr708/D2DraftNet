import os
import time
import click
import requests
import pandas as pd
from d2draftnet.config import MATCH_DATA_PATH, KEY
from d2draftnet.collect_match_ids import request_match_ids  # Import your match ID collector

def get_api_config(key):
    """
    Return API configuration based on the provided key.
    
    - Free Tier (key is None):
         * Call Limit: 2000 per day
         * Rate Limit: 60 calls per minute
    - Premium Tier (key provided):
         * Call Limit: Unlimited
         * Rate Limit: 1200 calls per minute
    """
    if key is None:
        return {"call_limit": 2000, "rate_limit": 60}
    else:
        return {"call_limit": float('inf'), "rate_limit": 1200}

def load_existing_data(verbose=False):
    if os.path.exists(MATCH_DATA_PATH):
        if verbose:
            click.echo(f"Accessing parquet file at {MATCH_DATA_PATH}")
        return pd.read_parquet(MATCH_DATA_PATH)
    else:
        if verbose:
            click.echo(f"No parquet file found at {MATCH_DATA_PATH}, starting a new one.")
        return pd.DataFrame(columns=["match_id", "radiant_draft", "dire_draft", "winner", "match_type", "date"])

def save_data(df, verbose=False):
    if verbose:
        click.echo(f"Saving data to parquet file at {MATCH_DATA_PATH}")
    df.to_parquet(MATCH_DATA_PATH, index=False)

def get_open_dota_hero_map():
    url = "https://api.opendota.com/api/heroes"
    response = requests.get(url)
    if response.status_code == 200:
        heroes = response.json()
        return {hero["id"]: hero["localized_name"] for hero in heroes}
    else:
        click.echo("Error fetching hero map from OpenDota API")
        return {}

def get_match_details(match_id):
    url = f"https://api.opendota.com/api/matches/{match_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def extract_draft(match_data, hero_map):
    picks = match_data.get("picks_bans", [])
    radiant_ids = [entry["hero_id"] for entry in picks if entry.get("is_pick") and entry.get("team") == 0]
    dire_ids = [entry["hero_id"] for entry in picks if entry.get("is_pick") and entry.get("team") == 1]
    
    if len(radiant_ids) != 5 or len(dire_ids) != 5:
        return None, None, None
    
    radiant_picks = [hero_map.get(hero_id, str(hero_id)) for hero_id in radiant_ids]
    dire_picks = [hero_map.get(hero_id, str(hero_id)) for hero_id in dire_ids]
    winner = "Radiant" if match_data.get("radiant_win") else "Dire"
    return radiant_picks, dire_picks, winner

@click.group()
def cli():
    pass

@cli.command()
@click.option("--limit", default=10, help="Number of valid matches to process.")
@click.option("--match-type", default="pub", type=click.Choice(["pro", "pub"], case_sensitive=False),
              help="Type of match to process: 'pro' or 'pub'.")
@click.option("--pub-rank", default=30, type=int, help="Rank tier filter for public matches (only applies if match-type is 'pub').")
@click.option("--verbose", is_flag=True, help="Enable verbose debugging output.")
def collect(limit, match_type, pub_rank, verbose):
    """
    Collect match data from the OpenDota API using a pre-collected list of match IDs.
    """
    df_existing = load_existing_data(verbose)
    existing_match_ids = set(df_existing["match_id"]) if not df_existing.empty else set()
    
    api_config = get_api_config(KEY)
    delay = 60 / api_config["rate_limit"]
    hero_map = get_open_dota_hero_map()
    
    # Retrieve match IDs using the external function.
    if verbose:
        click.echo("Fetching match IDs using request_match_ids function.")
    match_ids = request_match_ids(limit, verbose=1)
    
    records = []
    api_calls = 0

    for match_id in match_ids:
        if api_calls >= api_config["call_limit"]:
            click.echo("Daily call limit reached.")
            break
        if match_id in existing_match_ids:
            if verbose:
                click.echo(f"Skipping match {match_id}: already exists.")
            continue

        match_data = get_match_details(match_id)
        api_calls += 1
        if not match_data or "picks_bans" not in match_data:
            if verbose:
                click.echo(f"Skipping match {match_id}: no picks_bans data.")
            continue

        if match_type.lower() == "pub" and pub_rank is not None:
            if match_data.get("rank_tier") != pub_rank:
                if verbose:
                    click.echo(f"Skipping match {match_id}: rank_tier {match_data.get('rank_tier')} does not equal {pub_rank}.")
                continue

        radiant_picks, dire_picks, winner = extract_draft(match_data, hero_map)
        if radiant_picks is None:
            if verbose:
                click.echo(f"Skipping match {match_id}: invalid draft (not exactly 5 picks per team).")
            continue

        start_time = match_data.get("start_time")
        match_date = pd.to_datetime(start_time, unit="s") if start_time else None

        records.append({
            "date": match_date,
            "match_id": match_id,
            "match_type": match_type.lower(),
            "winner": winner,
            "radiant_draft": radiant_picks,
            "dire_draft": dire_picks,
        })
        if verbose:
            found = len(records)
            percent = found / limit * 100
            click.echo(f"Progress: {found}/{limit} valid matches found ({percent:.1f}%).")
        time.sleep(delay)
        if len(records) >= limit:
            break
    
    if records:
        df_new = pd.DataFrame(records)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["match_id"])
        save_data(df_combined, verbose)
        if len(records) < limit:
            click.echo(f"Only {len(records)} valid records found from available matches. Total records: {len(df_combined)}")
        else:
            click.echo(f"Added {len(records)} new records. Total records: {len(df_combined)}")
    else:
        click.echo("No new records added.")

@cli.command()
@click.option("--rows", default=10, help="Number of rows to display from the Parquet file.")
def view(rows):
    """
    Load and display data from the match data Parquet file.
    """
    try:
        df = pd.read_parquet(MATCH_DATA_PATH)
    except Exception as e:
        click.echo(f"Error reading Parquet file: {e}")
        return

    click.echo(df.head(rows).to_string(index=False))

if __name__ == "__main__":
    cli()
