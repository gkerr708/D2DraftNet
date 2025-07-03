import requests
from bs4 import BeautifulSoup
from typing import List
import time

"""
This works.
"""

def request_match_ids(N_ids: int, verbose: int = 0) -> List[str]:

    base_url = "https://www.dotabuff.com/matches?page={}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    all_match_ids = []
    ids_per_page = 100    
    pages_to_scrape = N_ids // ids_per_page + 1

    
    for page in range(1, pages_to_scrape + 1):
        url = base_url.format(page)
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
    
            # Extract match IDs from links that begin with '/matches/'
            match_ids = [a["href"].split("/")[-1] for a in soup.select("a[href^='/matches/']")]
            all_match_ids.extend(match_ids)
            if verbose > 1:
                print(f"Page {page}: {len(match_ids)} match IDs extracted.")
        else:
            if verbose > 1:
                print(f"Error: Unable to retrieve page {page}.")
        
        time.sleep(1)  # Delay to prevent overloading the server

    unique_match_ids = list(set(all_match_ids))

    if verbose:
        print(f"Found {len(unique_match_ids)} unique match IDs.")

    return unique_match_ids


def main(N_matches: int):
    base_url = "https://www.dotabuff.com/matches?page={}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    ids_per_page = 100    
    pages_to_scrape = N_matches // ids_per_page + 1

    for page in range(1, pages_to_scrape + 1):
        url = base_url.format(page)
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for table in soup.find_all("tr"):
                for th in table.find_all("th"):
                    # Print the plain text content.
                    print(th.find_all("div"))
                exit()


    

if __name__ == "__main__":
    main(1)




    
