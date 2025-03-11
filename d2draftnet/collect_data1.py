import requests
from dataclasses import dataclass
from config import KEY
from pathlib import Path
import pandas as pd
import click

"""
Gets match data from the OpenDota API and stores it in Parquet format.
Strategy: Use the OpenDota API call: /publicMatches to get public match data.
"""

@dataclass
class DataFetcher:
    base_url: str  # The base URL for the OpenDota API.

    def __post_init__(self):
        self.hero_id_to_name = self._openDotaID_2_heroname()
        self.output_path = Path("match_data.parquet")
        # Load existing data if the Parquet file exists, otherwise initialize an empty list.
        if self.output_path.exists():
            self.match_data = pd.read_parquet(self.output_path).to_dict(orient="records")
            click.secho(
                f"Importing dataset from {self.output_path},", fg="yellow"
            )
            click.secho(
                f"with {len(self.match_data)} matches, appending new data.", fg="yellow"
            )
        else:
            self.match_data = []
            click.secho("No existing dataset found, creating new one.", fg="green")

    def _openDotaID_2_heroname(self) -> dict:
        """
        Converts the OpenDota hero ID to the hero name.
        """
        response = requests.get(f"{self.base_url}/heroes?api_key={KEY}").json()
        return {hero['id']: hero['localized_name'] for hero in response}

    def _filter_by_rank(self, rank_tier: int) -> dict:
        """
        Filters the match data by the rank tier.
        """
        raise NotImplementedError

    def _get_pub_data(self, use_key=False) -> list:
        """
        Retrieves public match data using the API key.
        """
        if use_key:
            response = requests.get(f"{self.base_url}/publicMatches?api_key={KEY}")
        else:
            response = requests.get(f"{self.base_url}/publicMatches")
        if response.status_code == 200:
            return response.json()
        return None

    def _write_to_parquet(self, new_match_data: list) -> None:
        """
        Appends new match data to the existing dataset, deduplicates by match_id,
        and writes it to a Parquet file.
        """
        self.match_data.extend(new_match_data)
        # Deduplicate using match_id as the key.
        unique_matches = {match['match_id']: match for match in self.match_data}
        self.match_data = list(unique_matches.values())
        df = pd.DataFrame(self.match_data)
        df.to_parquet(self.output_path, index=False)
        click.secho(
            f"Saved {len(new_match_data)} new matches. Total dataset now has {len(self.match_data)} matches.",
            fg="blue"
        )

    def checkout_dataset(self) -> None:
        """
        Checks out the dataset.
        """
        if self.match_data:
            ids = [match['match_id'] for match in self.match_data]
            click.secho(f"Duplicates: {len(ids) - len(set(ids))}", fg="cyan")
        else:
            click.secho("Dataset is empty.", fg="cyan")

    def _df_to_dict(self) -> dict:
        """
        Converts the current dataset (list of dicts) into a dictionary mapping match_id to match info.
        """
        return {match['match_id']: match for match in self.match_data}

    def _is_duplicate(self, new_match_id, new_matches: list) -> bool:
        """
        Checks if a match with new_match_id already exists in either the stored dataset or the new_matches list.
        """
        if any(match['match_id'] == new_match_id for match in self.match_data):
            return True
        if any(match['match_id'] == new_match_id for match in new_matches):
            return True
        return False

    def analyze_pub_data(self, N_matches: int, verbose: bool = False) -> None:
        """
        Collects public match data until N_matches unique matches are acquired.
        """
        new_match_data = []
        collected = 0
        duplicates = 0
        with click.progressbar(length=N_matches, label="Collecting new matches") as bar:
            while collected < N_matches:
                pub_data = self._get_pub_data()  # List of public matches
                if pub_data is None:
                    click.secho("No public data retrieved", fg="red")
                    return

                for match in pub_data:
                    if collected >= N_matches:
                        break
                    if self._is_duplicate(match["match_id"], new_match_data):
                        duplicates += 1
                        continue

                    radiant_draft = [self.hero_id_to_name.get(hero_id, f"Unknown({hero_id})")
                                     for hero_id in match.get("radiant_team", [])]
                    dire_draft = [self.hero_id_to_name.get(hero_id, f"Unknown({hero_id})")
                                  for hero_id in match.get("dire_team", [])]
                    match_info = {
                        "match_id": match["match_id"],
                        "radiant_draft": radiant_draft,
                        "dire_draft": dire_draft,
                        "duration": match["duration"],
                        "winner": "Radiant" if match["radiant_win"] else "Dire"
                    }
                    new_match_data.append(match_info)
                    collected += 1
                    bar.update(1)
        self._write_to_parquet(new_match_data)
        if verbose:
            click.secho(f"Found {duplicates} duplicates.", fg="yellow")


if __name__ == "__main__":
    base_url = "https://api.opendota.com/api"
    fetcher = DataFetcher(base_url)
    fetcher.analyze_pub_data(
        N_matches=100, 
        verbose=True
    )
    #fetcher.checkout_dataset()
