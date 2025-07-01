# fbref_match_stats_scraper.py
import os
import time
import pandas as pd
import random
import requests
from bs4 import BeautifulSoup

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]

HEADERS = {'User-Agent': random.choice(USER_AGENTS)}

STAT_TYPES = {
    'shooting': 'Shooting',
    'keeper': 'Keeper',
    'passing': 'Passing',
    'gca': 'GCA',
    'possession': 'Possession'
}

def scrape_team_match_logs(team_id, team_name, season, stat_type):
    stat_name = STAT_TYPES[stat_type]
    url = f"https://fbref.com/en/squads/{team_id}/{season}/matchlogs/c9/{stat_type}/{team_name.replace(' ', '-')}-Match-Logs-Premier-League"
    print(f"Scraping {stat_type.title()} for {team_name} ({season}): {url}")

    res = requests.get(url, headers=HEADERS)
    if res.status_code != 200:
        print(f"Failed to load {url}")
        return pd.DataFrame()

    soup = BeautifulSoup(res.content, 'html.parser')
    table = soup.find('table')
    if table is None:
        print(f"No table for {stat_type} at {url}")
        return pd.DataFrame()

    df = pd.read_html(str(table))[0]
    df = df[df['Comp'] == 'Premier League']

    df['team'] = team_name
    df['stat_type'] = stat_type
    df['season'] = f"{season}-{int(season)+1}"
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    return df

def merge_stats(fixture_df, stats_df, team_col, side):
    merged = fixture_df.copy()
    for stat_type, df in stats_df.items():
        if df.empty:
            continue
        stat_cols = [c for c in df.columns if c not in ['Rk', 'Date', 'Opponent', 'Comp', 'team', 'stat_type', 'season']]
        df = df.rename(columns={c: f"{side}_{stat_type}_{c}" for c in stat_cols})
        merged = pd.merge(
            merged,
            df[['Date'] + stat_cols + ['team']],
            left_on=['date', team_col],
            right_on=['Date', 'team'],
            how='left'
        )
        merged = merged.drop(columns=['Date', 'team'])
    return merged
