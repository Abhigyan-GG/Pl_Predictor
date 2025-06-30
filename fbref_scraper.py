import os
import time
import pandas as pd
import random
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ========== Setup Paths ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'get_data', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ========== Constants ==========
SEASONS = list(range(2014, 2025))  # From 2014-15 to 2024-25 seasons
BASE_URL = "https://fbref.com/en/comps/9/Premier-League-Stats"

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def get_season_url(season):
    """Get the correct URL for a specific season"""
    if season == 2024:  # Current season 2024-25
        return BASE_URL
    else:
        # Historical seasons use a different URL pattern
        season_str = f"{season}-{season+1}"
        return f"https://fbref.com/en/comps/9/{season_str}/{season_str}-Premier-League-Stats"

def scrape_with_requests(url, session=None, max_retries=3):
    """Alternative scraping method using requests instead of playwright"""
    if session is None:
        session = requests.Session()
    
    headers = {
        'User-Agent': get_random_user_agent(),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0'
    }
    
    for attempt in range(max_retries):
        try:
            print(f"🌐 Requests attempt {attempt + 1}: {url}")
            time.sleep(random.uniform(2, 5))
            
            response = session.get(url, headers=headers, timeout=30, verify=False)
            response.raise_for_status()
            
            if response.status_code == 200:
                return BeautifulSoup(response.content, 'html.parser')
            else:
                print(f"⚠️ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Requests attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(random.uniform(5, 10))
    
    return None

def scrape_with_playwright_simple(url, max_retries=2):
    """Simplified playwright approach"""
    for attempt in range(max_retries):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(user_agent=get_random_user_agent())
                page = context.new_page()
                
                print(f"🎭 Playwright attempt {attempt + 1}: {url}")
                
                # Much simpler approach - don't wait for networkidle
                page.goto(url, timeout=45000)
                time.sleep(random.uniform(3, 6))
                
                content = page.content()
                browser.close()
                
                return BeautifulSoup(content, 'html.parser')
                
        except Exception as e:
            print(f"❌ Playwright attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(random.uniform(5, 10))
    
    return None

def find_fixtures_link(soup, season):
    """Find the correct fixtures link for the specific season"""
    fixtures_link = None
    
    # Method 1: Look for specific Premier League schedule link
    for link in soup.find_all('a', href=True):
        href = link.get('href', '')
        text = link.text.strip().lower()
        
        # Look for schedule/fixture links that contain the season info
        if '/comps/9/schedule' in href or '/comps/9/fixtures' in href:
            if f"{season}-{season+1}" in href or (season == 2024 and 'premier-league-scores' in href.lower()):
                return href
        elif 'scores & fixtures' in text and '/comps/9/' in href:
            return href
    
    # Method 2: Construct the URL based on season
    if season == 2024:  # Current season
        return "/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
    else:
        season_str = f"{season}-{season+1}"
        return f"/en/comps/9/schedule/{season_str}/{season_str}-Premier-League-Scores-and-Fixtures"

def scrape_season_matches(season_url, season, max_retries=2):
    print(f"\n🔍 Scraping matches for season {season}-{season+1}")
    matches_data = []
    
    # Try requests first (faster and more reliable)
    print("🚀 Trying requests method...")
    session = requests.Session()
    soup = scrape_with_requests(season_url, session)
    
    if not soup:
        print("🚀 Trying playwright method...")
        soup = scrape_with_playwright_simple(season_url)
    
    if not soup:
        print("❌ Could not fetch season page with any method")
        return []
    
    # Find fixtures link
    fixtures_link = find_fixtures_link(soup, season)
    if fixtures_link.startswith('/'):
        fixtures_url = f"https://fbref.com{fixtures_link}"
    else:
        fixtures_url = fixtures_link
    
    print(f"📄 Fetching fixtures from: {fixtures_url}")
    
    # Get fixtures page
    fixtures_soup = scrape_with_requests(fixtures_url, session)
    if not fixtures_soup:
        fixtures_soup = scrape_with_playwright_simple(fixtures_url)
    
    if not fixtures_soup:
        print("❌ Could not fetch fixtures page")
        return []
    
    # Find matches table
    matches_table = None
    
    print("🔍 Looking for matches table...")
    all_tables = fixtures_soup.find_all('table')
    print(f"📊 Found {len(all_tables)} tables on page")
    
    # Try different approaches to find the table
    table_patterns = [
        {'id': lambda x: x and 'fixtures' in x},
        {'id': lambda x: x and 'sched' in x},
        {'class': lambda x: x and any(k in str(x) for k in ['fixtures', 'schedule'])}
    ]
    
    for i, pattern in enumerate(table_patterns):
        matches_table = fixtures_soup.find('table', pattern)
        if matches_table:
            print(f"✅ Found matches table using pattern {i+1}")
            break
    
    # Debug: show all table info
    if not matches_table:
        print("🔍 Checking all tables for match data...")
        for i, table in enumerate(all_tables):
            table_id = table.get('id', 'no-id')
            table_class = table.get('class', ['no-class'])
            
            # Check headers
            headers = [th.text.strip().lower() for th in table.find_all('th')]
            sample_cells = []
            first_row = table.find('tr')
            if first_row:
                sample_cells = [td.text.strip()[:15] for td in first_row.find_all(['td', 'th'])]
            
            print(f"   Table {i+1}: id='{table_id}', class='{table_class}'")
            print(f"      Headers: {headers[:6]}")
            print(f"      Sample: {sample_cells[:6]}")
            
            # Look for fixture-like content
            if any(keyword in ' '.join(headers) for keyword in ['date', 'home', 'away', 'score']):
                matches_table = table
                print(f"✅ Selected table {i+1} based on headers")
                break
    
    if not matches_table:
        print("⚠️ Could not find matches table")
        return []
    
    # Debug: Show table structure first
    print("🔍 Analyzing table structure...")
    sample_rows = matches_table.find_all('tr')[:5]
    for i, row in enumerate(sample_rows):
        cells = row.find_all(['th', 'td'])
        cell_texts = [cell.text.strip()[:20] for cell in cells]
        print(f"   Row {i}: {len(cells)} cells -> {cell_texts}")
    
    # Parse matches
    rows_processed = 0
    for row_idx, row in enumerate(matches_table.find_all('tr')):
        # Skip header rows
        if row.find('th', {'scope': 'col'}):
            continue
            
        cells = row.find_all(['th', 'td'])
        if len(cells) < 4:
            continue
        
        try:
            # Debug first few rows
            if row_idx < 10:
                cell_texts = [cell.text.strip() for cell in cells]
                print(f"🔍 Row {row_idx}: {cell_texts}")
            
            date_cell = cells[0].text.strip()
            if not date_cell or date_cell.lower() in ['date', 'wk', 'week', 'round']:
                continue
            
            # Standard FBRef fixtures table structure:
            # 0: Date, 1: Time, 2: Home, 3: Score, 4: Away, 5: Venue, 6: Referee, 7: Match Report, 8: Notes
            if len(cells) >= 5:
                home_team = cells[2].text.strip()
                score = cells[3].text.strip()
                away_team = cells[4].text.strip()
                
                # Validate that we have actual team names
                if (home_team and away_team and 
                    len(home_team) > 1 and len(away_team) > 1 and
                    home_team.lower() not in ['home', 'team', ''] and
                    away_team.lower() not in ['away', 'team', '']):
                    
                    match_data = {
                        'season': f"{season}-{season+1}",
                        'date': date_cell,
                        'time': cells[1].text.strip() if len(cells) > 1 else '',
                        'home_team': home_team,
                        'score': score,
                        'away_team': away_team,
                        'venue': cells[5].text.strip() if len(cells) > 5 else '',
                        'competition': 'Premier League',
                    }
                    
                    matches_data.append(match_data)
                    rows_processed += 1
                    
                    # Debug: show first few matches found
                    if rows_processed <= 3:
                        print(f"✅ Match {rows_processed}: {home_team} vs {away_team} ({score})")
                    
        except Exception as e:
            if row_idx < 10:  # Only show errors for first few rows
                print(f"⚠️ Error parsing row {row_idx}: {e}")
            continue
    
    print(f"✅ Found {len(matches_data)} Premier League matches for {season}-{season+1}")
    return matches_data

def parse_score(score_text):
    """Parse score text into home and away goals"""
    try:
        if not score_text or score_text.strip() == '':
            return pd.NA, pd.NA
            
        for separator in ['–', '−', '-', ':']:
            if separator in score_text:
                parts = score_text.split(separator)
                if len(parts) == 2:
                    home_goals = int(parts[0].strip())
                    away_goals = int(parts[1].strip())
                    return home_goals, away_goals
        
        return pd.NA, pd.NA
    except (ValueError, AttributeError):
        return pd.NA, pd.NA

def determine_result(row):
    """Determine match result based on goals"""
    if pd.isna(row['home_goals']) or pd.isna(row['away_goals']):
        return 'Unknown'
    if row['home_goals'] > row['away_goals']:
        return 'Home Win'
    elif row['home_goals'] < row['away_goals']:
        return 'Away Win'
    else:
        return 'Draw'

def main():
    print("🚀 Starting FBRef Premier League scraper...")
    print(f"📅 Seasons to scrape: {SEASONS}")
    print("💡 Using hybrid approach: requests + playwright fallback")
    print("🔧 Fixed URL generation for historical seasons")
    
    all_matches = []

    for i, season in enumerate(SEASONS):
        season_url = get_season_url(season)
        print(f"\n🌐 Processing season {season}-{season+1}")
        print(f"🔗 URL: {season_url}")
        
        if i > 0:
            delay = random.uniform(5, 10)
            print(f"⏳ Waiting {delay:.1f} seconds between seasons...")
            time.sleep(delay)

        season_matches = scrape_season_matches(season_url, season)
        all_matches.extend(season_matches)
        print(f"📈 Completed season {season}-{season+1}, total matches so far: {len(all_matches)}")

    # Process and save data
    if all_matches:
        print(f"\n📊 Processing {len(all_matches)} matches...")
        df = pd.DataFrame(all_matches)

        df[['home_goals', 'away_goals']] = df['score'].apply(lambda x: pd.Series(parse_score(x)))
        df['result'] = df.apply(determine_result, axis=1)

        df = df.drop_duplicates()
        df = df.sort_values(['season', 'date']).reset_index(drop=True)

        output_path = os.path.join(DATA_DIR, 'fbref_premier_league_matches.csv')
        df.to_csv(output_path, index=False)
        
        print(f"\n💾 Data saved to: {output_path}")
        print(f"✅ Total matches scraped: {len(df)}")
        print(f"📈 Matches with scores: {len(df[~pd.isna(df['home_goals'])])}")
        print(f"🏆 Results distribution:")
        print(df['result'].value_counts())
        
        # Show season breakdown
        print(f"\n📋 Matches per season:")
        season_counts = df['season'].value_counts().sort_index()
        for season, count in season_counts.items():
            print(f"   {season}: {count} matches")
        
        print(f"\n📋 Sample of scraped data:")
        print(df.head(3).to_string(index=False))
        
    else:
        print("\n⚠️ No data was collected.")
        print("🔍 Troubleshooting suggestions:")
        print("   1. Try changing SEASONS to [2023] for last season")
        print("   2. Check if you can access fbref.com in your browser")
        print("   3. Try running with a VPN if your IP is blocked")
        print("   4. Run during off-peak hours")

if __name__ == "__main__":
    main()