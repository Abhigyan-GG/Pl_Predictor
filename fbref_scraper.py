import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
import random

# ========== Setup Paths ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'get_data', 'data')
os.makedirs(DATA_DIR, exist_ok=True)  # create data dir if it doesn't exist

# ========== Constants ==========
SEASONS = list(range(2019, 2024))  # 2019-20 to 2023-24 seasons
BASE_URL = "https://fbref.com/en/comps/9/Premier-League-Stats"

# User agent rotation to avoid blocking
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/109.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
]

# ========== Functions ==========
def get_random_headers():
    """Generate random headers for requests to mimic regular browser traffic"""
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0',
    }

def scrape_season_matches(season_url, season):
    """Scrape all matches for a given season"""
    print(f"Scraping matches for season {season}")
    
    # Get the season page
    try:
        response = requests.get(season_url, headers=get_random_headers(), timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching season page: {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Look for the link to the scores & fixtures page
    fixtures_link = None
    for link in soup.find_all('a'):
        if 'Scores & Fixtures' in link.text:
            fixtures_link = link.get('href')
            break
    
    if not fixtures_link:
        print(f"Could not find fixtures link for season {season}")
        return []
    
    fixtures_url = f"https://fbref.com{fixtures_link}"
    print(f"Fetching fixtures from: {fixtures_url}")
    
    # Get the fixtures page
    try:
        # Add a random delay to be respectful and avoid getting blocked
        time.sleep(random.uniform(2, 5))
        response = requests.get(fixtures_url, headers=get_random_headers(), timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching fixtures page: {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the matches table
    matches_table = soup.find('table', id=lambda x: x and 'fixtures' in x)
    if not matches_table:
        print(f"Could not find matches table for season {season}")
        return []
    
    # Extract matches data
    matches_data = []
    
    for row in matches_table.find_all('tr'):
        # Skip header rows
        if row.find('th', {'scope': 'col'}):
            continue
        
        cells = row.find_all(['th', 'td'])
        if len(cells) < 8:  # Ensure we have enough columns
            continue
        
        try:
            date_cell = cells[0].text.strip()
            # Skip rows without a date (might be headers or separators)
            if not date_cell or date_cell == '':
                continue
                
            # Extract match info
            match_data = {
                'season': f"{season}-{season+1}",
                'date': date_cell,
                'time': cells[1].text.strip(),
                'home_team': cells[2].text.strip(),
                'score': cells[3].text.strip(),
                'away_team': cells[4].text.strip(),
                'venue': cells[5].text.strip(),
                'competition': cells[6].text.strip(),
            }
            
            # Only include Premier League matches
            if 'Premier League' in match_data['competition']:
                matches_data.append(match_data)
                
        except Exception as e:
            print(f"Error parsing row: {e}")
            continue
    
    print(f"Found {len(matches_data)} Premier League matches for season {season}-{season+1}")
    return matches_data

# ========== Main Scraping Process ==========
all_matches = []

for season in SEASONS:
    # Construct the URL for the season
    if season == max(SEASONS):  # Current season
        season_url = BASE_URL
    else:
        season_url = f"{BASE_URL}-{season}-{season+1}"
    
    print(f"Processing URL: {season_url}")
    
    # Add a random delay between seasons to be respectful to the server
    if season != SEASONS[0]:
        time.sleep(random.uniform(3, 8))
        
    # Scrape the season's matches
    season_matches = scrape_season_matches(season_url, season)
    all_matches.extend(season_matches)
    
    print(f"Completed season {season}-{season+1}, total matches so far: {len(all_matches)}")

# ========== Save Data ==========
if all_matches:
    df = pd.DataFrame(all_matches)
    
    # Parse score column to extract home and away goals
    def parse_score(score_text):
        try:
            if '–' in score_text:  # This is an en dash
                parts = score_text.split('–')
            elif '-' in score_text:  # Regular hyphen
                parts = score_text.split('-')
            else:
                return pd.NA, pd.NA
                
            return int(parts[0].strip()), int(parts[1].strip())
        except:
            return pd.NA, pd.NA
    
    # Add home and away goals columns
    df['home_goals'], df['away_goals'] = zip(*df['score'].apply(parse_score))
    
    # Add result column (from home team's perspective)
    def determine_result(row):
        if pd.isna(row['home_goals']) or pd.isna(row['away_goals']):
            return 'Unknown'
        if row['home_goals'] > row['away_goals']:
            return 'Home Win'
        elif row['home_goals'] < row['away_goals']:
            return 'Away Win'
        else:
            return 'Draw'
    
    df['result'] = df.apply(determine_result, axis=1)
    
    # Save to CSV
    output_path = os.path.join(DATA_DIR, 'fbref_premier_league_matches.csv')
    df.to_csv(output_path, index=False)
    print(f'Data saved to {output_path}')
    print(f'Total matches scraped: {len(df)}')
else:
    print("No data was collected. Check for possible website structure changes or blocking.")