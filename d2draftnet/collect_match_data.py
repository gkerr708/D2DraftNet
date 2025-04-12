import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import date
from tqdm import tqdm  # For progress bar

from d2draftnet.config import MATCH_DATA_PATH

def main(N_matches: int):
    base_url = "https://www.dotabuff.com/matches?page={}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    matches_list = []

    # Load existing dataset and create a set of existing match IDs
    parquet_file = MATCH_DATA_PATH
    existing_ids = set()
    if parquet_file.exists():
        df_existing = pd.read_parquet(parquet_file)
        existing_ids = set(df_existing['id'].dropna().astype(str))
    
    page = 1
    with tqdm(total=N_matches, desc="Scraping Matches", unit="match") as pbar:
        while len(matches_list) < N_matches:
            url = base_url.format(page)
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                print(f"Error: status code {response.status_code} for page {page}")
                break

            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table")
            if table:
                rows = table.find_all("tr")
                # Skip header row
                for tr in rows[1:]:
                    tds = tr.find_all("td")
                    if len(tds) < 6:
                        continue

                    td0 = tds[0]
                    match_id = td0.find("a").get_text(strip=True) if td0.find("a") else None
                    if not match_id or match_id in existing_ids:
                        continue

                    match_date = td0.find("time").get_text(strip=True) if td0.find("time") else str(date.today())
                    
                    td1 = tds[1]
                    game_mode = td1.contents[0].strip() if td1.contents else ""
                    skill_div = td1.find("div", class_="subtext")
                    skill = skill_div.get_text(strip=True) if skill_div else ""
                    
                    td2 = tds[2]
                    result_a = td2.find("a")
                    result = result_a.get_text(strip=True) if result_a else td2.get_text(strip=True)
                    
                    td3 = tds[3]
                    duration = td3.contents[0].strip() if td3.contents else ""
                    
                    td4 = tds[4]
                    radiant_draft = [img.get("title") for img in td4.find_all("img") if img.get("title")]
                    
                    td5 = tds[5]
                    dire_draft = [img.get("title") for img in td5.find_all("img") if img.get("title")]
                    
                    match_data = {
                        "id": match_id,
                        "date": match_date,
                        "duration": duration,
                        "result": result,
                        "game_mode": game_mode,
                        "skill": skill,
                        "radiant_draft": radiant_draft,
                        "dire_draft": dire_draft
                    }
                    
                    matches_list.append(match_data)
                    existing_ids.add(match_id)
                    pbar.update(1)
                    
                    if len(matches_list) >= N_matches:
                        break
            page += 1

    df_new = pd.DataFrame(matches_list)
    
    if parquet_file.exists():
        if df_new.empty:
            print("No new matches to add.")
        else:
            df_existing = pd.read_parquet(parquet_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_parquet(parquet_file, index=False)
            print(f"Appended {len(df_new)} new matches to {parquet_file}.")
    else:
        df_new.to_parquet(parquet_file, index=False)
        print(f"Saved {len(df_new)} matches to {parquet_file}.")

    print(f"Total matches: {len(existing_ids)}")

if __name__ == "__main__":
    main(10_000)
