import requests
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from config import MATCH_DATA_PATH, HEROS
import click

key = '21bcc3a2-28fc-4331-a1ad-5e68401b8f71'

@dataclass
class MatchData:
    match_id: int
    radiant_team: str
    dire_team: str
    duration: float
    winner: str
    match_type: str
    match_date: str
    radiant_vector: list
    dire_vector: list

@dataclass
class Dota2MatchFetcher:
    output_csv: Path
    num_pro_matches: int = 0
    num_pub_matches: int = 100
    hero_id_to_index: dict = field(default_factory=dict)
    hero_id_to_name: dict = field(default_factory=dict)
    total_heroes: int = 0
    match_ids: list[int] = field(default_factory=list, init=False)
    existing_match_ids: set[int] = field(default_factory=set, init=False)

    def fetch_hero_data(self, verbose: bool = False) -> None:
        """Fetch hero data from OpenDota API. Matches OpenDota hero IDs to alphanumeric hero name indices."""
        click.secho("\nFetching hero data from OpenDota API...", fg="yellow")
        heroes_url = f"https://api.opendota.com/api/heroes?api_key={key}"
        response = requests.get(heroes_url)

        if response.status_code == 200:
            heroes_data = response.json()

            self.total_heroes = len(HEROS)
            N_heros_identified = 0
            for hero in heroes_data:
                for hero_index, hero_name in enumerate(HEROS):
                    if hero_name == hero['localized_name']:
                        self.hero_id_to_index[hero['id']] = hero_index
                        self.hero_id_to_name[hero['id']] = hero_name
                        N_heros_identified += 1
                        if verbose:
                            print(f"Hero ID: {hero['id']}, Hero Name: {hero['localized_name']}")

            if N_heros_identified != self.total_heroes:
                raise click.ClickException("Error: Not all heroes were identified.")
            if verbose:
                print(f"Identified {N_heros_identified} heroes out of {self.total_heroes}.")

            click.secho("Successfully fetched hero data.", fg="green")
        else:
            raise click.ClickException("Error fetching hero names from OpenDota API.")

    def initialize_csv(self) -> None:
        """Create the output CSV file if it doesn't exist."""
        if not self.output_csv.exists():
            click.secho(f"\nOutput CSV file not found. Creating new CSV file: {self.output_csv}", fg="yellow")
            with open(self.output_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                match_id_header = ["Match_ID"]
                radiant_header = [f"Radiant_Hero_{i}" for i in range(5)]
                dire_header = [f"Dire_Hero_{i}" for i in range(5)]
                metadata_header = ["Radiant Team", "Dire Team", "Match Duration (mins)", "Winner", "Match Type", "Match Date"]
                header = match_id_header + metadata_header + radiant_header + dire_header
                writer.writerow(header)
            click.secho("CSV file created successfully.", fg="green")
        else:
            click.secho(f"\nAppending new matches to existing file: {self.output_csv}...", fg="green")

    def load_existing_match_ids(self) -> None:
        """Load existing match IDs from the CSV."""
        if self.output_csv.exists():
            df = pd.read_csv(self.output_csv)
            if "Match_ID" in df.columns:
                self.existing_match_ids = set(df["Match_ID"])

    def fetch_match_ids(self, url, match_type, limit, min_rank=None, max_rank=None) -> list:
        """Fetch match IDs from OpenDota API with optional rank filtering."""
        click.secho(f"\nFetching {limit} {match_type} match IDs...", fg="yellow")
        match_ids = []
        retries = 0
        max_retries = 10
        backoff_time = 2
        last_match_id = None

        with tqdm(total=limit, desc=f"Fetching {match_type} Match IDs") as pbar:
            while len(match_ids) < limit:
                params = {
                    "api_key": key,
                    "less_than_match_id": last_match_id,
                    "mmr_descending": 1,
                    "min_rank": min_rank,
                    "max_rank": max_rank,
                }
                response = requests.get(url, params={k: v for k, v in params.items() if v is not None})
                if response.status_code == 200:
                    matches = response.json()
                    if not matches:
                        break

                    for match in matches:
                        if len(match_ids) >= limit:
                            break
                        match_id = match['match_id']
                        if match_id not in self.existing_match_ids:
                            match_ids.append(match_id)
                            self.existing_match_ids.add(match_id)
                            last_match_id = match_id
                            pbar.update(1)

                elif response.status_code == 429:
                    if retries >= max_retries:
                        click.secho("Exceeded maximum retries. Exiting...", fg="red")
                        break
                    click.secho(f"Rate limit reached. Retrying in {backoff_time} seconds...", fg="yellow")
                    time.sleep(backoff_time)
                    backoff_time *= 2
                    retries += 1
                else:
                    click.secho(f"Error fetching {match_type} match IDs: {response.status_code}", fg="red")
                    break

        click.secho(f"Fetched {len(match_ids)} {match_type} match IDs.", fg="green")
        return match_ids

    def fetch_and_save_match_data(self, match_ids, match_type, verbose=True):
        """Fetch detailed match data and save to the CSV using a DataFrame."""
        match_data_list = []

        BASE_URL = "https://api.opendota.com/api"
        N_drafts_found = 0

        with tqdm(total=len(match_ids), desc=f"Fetching {match_type} Match Data") as pbar:
            for match_id in match_ids:
                match_detail_url = f"{BASE_URL}/matches/{match_id}?api_key={key}"
                response = requests.get(match_detail_url)

                if response.status_code == 200:
                    match_data = response.json()

                    winner = "Radiant" if match_data['radiant_win'] else "Dire"
                    start_time = match_data.get('start_time')
                    match_date = datetime.utcfromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S') if start_time else 'Unknown Date'

                    radiant_draft = []
                    dire_draft = []

                    if 'picks_bans' in match_data and match_data['picks_bans'] is not None:
                        for pick_ban in match_data['picks_bans']:
                            if pick_ban['is_pick']:
                                hero_id = pick_ban['hero_id']
                                team = pick_ban['team']
                                hero_index = self.hero_id_to_index.get(hero_id)
                                name = self.hero_id_to_name.get(hero_id)
                                if hero_index is not None:
                                    if team == 0:
                                        radiant_draft.append(name)
                                    else:
                                        dire_draft.append(name)

                    if len(radiant_draft) == 5 and len(dire_draft) == 5:
                        match_data_list.append({
                            "Match ID": match_id,
                            "Match Date": match_date,
                            "Match Type": match_type,
                            "Winner": winner,
                            **{f"Radiant_Hero_{i}": radiant_draft[i] for i in range(5)},
                            **{f"Dire_Hero_{i}": dire_draft[i] for i in range(5)}
                        })
                        N_drafts_found += 1

                    pbar.update(1)

                elif response.status_code == 429:
                    click.secho("Rate limit reached. Retrying...", fg="yellow")
                    time.sleep(2)
                else:
                    click.secho(f"Error fetching match {match_id}: {response.status_code}", fg="red")

        # Makes sure at least one draft was found - else there are errors
        if N_drafts_found == 0:
            click.secho(f"No complete drafts found for {match_type} matches.", fg="red")
            return

        # Converts the list of match data dictionaries to a DataFrame
        match_df = pd.DataFrame(match_data_list)

        # Check to see if the csv already exists we append to it 
        if self.output_csv.exists():
            try:
                existing_df = pd.read_csv(self.output_csv)
                match_df = pd.concat([existing_df, match_df], ignore_index=True)
            except pd.errors.EmptyDataError:
                print("Empty data error.")
                print("Try deleting it.")
                exit()

        # Save the DataFrame to the CSV
        match_df.to_csv(self.output_csv, index=False)

        # Print out the results
        click.secho(f"Found {N_drafts_found} complete drafts out of {len(match_ids)} matches.", fg="green")
        click.secho(f"Match data saved to {self.output_csv}", fg="green")


# Debug function
def main():
    print()
    num_pub_matches = 100_000
    fetcher = Dota2MatchFetcher(output_csv=MATCH_DATA_PATH, num_pro_matches=0, num_pub_matches=num_pub_matches)
    fetcher.fetch_hero_data(verbose=False)
    ids = fetcher.fetch_match_ids(
        "https://api.opendota.com/api/publicMatches",
        "High-Level Pub",
        num_pub_matches,
        min_rank=50,
        max_rank=85
    )
    fetcher.fetch_and_save_match_data(ids, "High-Level Pub")

if __name__ == "__main__":
    main()
