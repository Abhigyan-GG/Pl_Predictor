import os
import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup

# ========== Setup Paths ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'get_data', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ========== Constants ==========
STARTING_URL = "https://fbref.com/en/comps/9/Premier-League-Stats"
SEASONS_TO_SCRAPE = 5  # How many seasons to go back (e.g., 5 seasons)
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/109.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
]

# ========== Functions ==========
def get_random_headers():
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }

def scrape_season(standings_url, year_label):
    """Scrape all teams for a given season"""
    print(f"Scraping season {year_label} from {standings_url}")
    
    try:
        response = requests.get(standings_url, headers=get_random_headers(), timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching standings page: {e}")
        return [], None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    try:
        standings_table = soup.select('table.stats_table')[0]
    except IndexError:
        print("Error: Could not find standings table.")
        return [], None
    
    links = [l.get("href") for l in standings_table.find_all('a') if l.get("href") and '/squads/' in l.get("href")]
    team_urls = [f"https://fbref.com{link}" for link in links]
    
    all_team_matches = []
    
    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        print(f"Scraping {team_name}...")
        
        try:
            team_response = requests.get(team_url, headers=get_random_headers(), timeout=30)
            team_response.raise_for_status()
        except Exception as e:
            print(f"Error fetching team page: {e}")
            continue
        
        try:
            matches = pd.read_html(team_response.text, match="Scores & Fixtures")[0]
        except ValueError:
            print(f"No Matches table found for {team_name}")
            continue
        
        team_soup = BeautifulSoup(team_response.text, 'html.parser')
        shooting_links = [l.get("href") for l in team_soup.find_all('a') if l.get("href") and 'all_comps/shooting/' in l.get("href")]
        
        if not shooting_links:
            print(f"No shooting stats found for {team_name}")
            continue
        
        shooting_url = f"https://fbref.com{shooting_links[0]}"
        
        try:
            shooting_response = requests.get(shooting_url, headers=get_random_headers(), timeout=30)
            shooting_response.raise_for_status()
            shooting = pd.read_html(shooting_response.text, match="Shooting")[0]
            shooting.columns = shooting.columns.droplevel()
        except Exception as e:
            print(f"Error fetching shooting stats: {e}")
            continue
        
        try:
            team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError:
            print(f"Error merging matches and shooting data for {team_name}")
            continue
        
        team_data = team_data[team_data["Comp"] == "Premier League"]
        
        team_data["Season"] = year_label
        team_data["Team"] = team_name
        all_team_matches.append(team_data)
        
        time.sleep(random.uniform(4, 6))  # polite pause
    
    # Get previous season URL
    try:
        previous_season_link = soup.select("a.prev")[0].get("href")
        previous_season_url = f"https://fbref.com{previous_season_link}"
    except (IndexError, AttributeError):
        print("Could not find link to previous season.")
        previous_season_url = None
    
    return all_team_matches, previous_season_url

# ========== Main Scraping ==========
all_seasons_data = []
current_url = STARTING_URL
current_year = 2023  # Start with 2023-24 season

for _ in range(SEASONS_TO_SCRAPE):
    season_label = f"{current_year-1}-{current_year}"
    season_data, next_url = scrape_season(current_url, season_label)
    
    all_seasons_data.extend(season_data)
    
    if not next_url:
        print("No more previous seasons found.")
        break
    
    current_url = next_url
    current_year -= 1
    time.sleep(random.uniform(8, 10))  # pause between seasons

# ========== Save Data ==========
if all_seasons_data:
    match_df = pd.concat(all_seasons_data)
    match_df.columns = [c.lower().replace(" ", "_") for c in match_df.columns]
    
    output_path = os.path.join(DATA_DIR, 'fbref_premier_league_matches_with_shooting.csv')
    match_df.to_csv(output_path, index=False)
    
    print(f"✅ Data scraping complete! Saved {len(match_df)} rows to {output_path}")
else:
    print("⚠️ No data collected. Please check scraping logic.")

