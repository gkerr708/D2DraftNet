import requests
from bs4 import BeautifulSoup

url = "https://www.dotabuff.com/matches"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)

soup = BeautifulSoup(response.text, "html.parser")
match_ids = [a["href"].split("/")[-1] for a in soup.select("a[href^='/matches/']")]

print(match_ids)  # Extracted match IDs
