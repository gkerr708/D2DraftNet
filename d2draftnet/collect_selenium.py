import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import date

from d2draftnet.config import MATCH_DATA_PATH

def main(N_matches: int):
    base_url = "https://www.dotabuff.com/matches?page={}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    ids_per_page = 100    
    pages_to_scrape = N_matches // ids_per_page + 1
    matches_list = []

    for page in range(1, pages_to_scrape + 1):
        url = base_url.format(page)
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table")
            if table:
                rows = table.find_all("tr")
                # Skip header row
                for tr in rows[1:]:
                    tds = tr.find_all("td")
                    if len(tds) >= 6:
                        # Extract match id and date from the first column.
                        td0 = tds[0]
                        match_id = td0.find("a").get_text(strip=True) if td0.find("a") else None
                        match_date = td0.find("time").get_text(strip=True) if td0.find("time") else str(date.today())
                        
                        # Extract game mode and skill from the second column.
                        td1 = tds[1]
                        game_mode = td1.contents[0].strip() if td1.contents else ""
                        skill_div = td1.find("div", class_="subtext")
                        skill = skill_div.get_text(strip=True) if skill_div else ""
                        
                        # Extract result from the third column.
                        td2 = tds[2]
                        result_a = td2.find("a")
                        result = result_a.get_text(strip=True) if result_a else td2.get_text(strip=True)
                        
                        # Extract duration from the fourth column.
                        td3 = tds[3]
                        duration = td3.contents[0].strip() if td3.contents else ""
                        
                        # Extract Radiant and Dire draft information from the fifth and sixth columns.
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

    # Convert to DataFrame and save as a Parquet file.
    parquet_file = MATCH_DATA_PATH
    df_new = pd.DataFrame(matches_list)
    
    if parquet_file.exists():
        df_existing = pd.read_parquet(parquet_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_parquet(parquet_file, index=False)
        print(f"Appended {len(df_new)} matches to {parquet_file}.")
    else:
        df_new.to_parquet(parquet_file, index=False)
        print(f"Saved {len(df_new)} matches to {parquet_file}.")

if __name__ == "__main__":
    main(100)
