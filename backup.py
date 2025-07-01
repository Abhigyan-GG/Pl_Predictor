import os
import time
import pandas as pd
import random
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import urllib3
from datetime import datetime
import re

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ========== Setup Paths ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'get_data', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ========== Constants ==========
SEASONS = list(range(2014, 2025))  # From 2014-15 to 2024-25 seasons

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def get_fixtures_url(season):
    """Get the direct fixtures URL for a specific season"""
    if season == 2024:  # Current season 2024-25
        return "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
    else:
        # Historical seasons - try the main season page first, then navigate to fixtures
        season_str = f"{season}-{season+1}"
        return f"https://fbref.com/en/comps/9/{season_str}/{season_str}-Premier-League-Stats"

def scrape_with_requests(url, session=None, max_retries=3):
    """Scraping method using requests"""
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
        'Referer': 'https://fbref.com/',
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

def scrape_with_playwright(url, max_retries=2):
    """Playwright scraping with better error handling"""
    for attempt in range(max_retries):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-dev-shm-usage']
                )
                context = browser.new_context(
                    user_agent=get_random_user_agent(),
                    viewport={'width': 1920, 'height': 1080}
                )
                page = context.new_page()
                
                print(f"🎭 Playwright attempt {attempt + 1}: {url}")
                
                # Navigate and wait for content
                page.goto(url, timeout=45000, wait_until='domcontentloaded')
                
                # Wait for tables to load
                try:
                    page.wait_for_selector('table', timeout=10000)
                except:
                    print("⚠️ No tables found, proceeding anyway")
                
                time.sleep(random.uniform(3, 6))
                
                content = page.content()
                browser.close()
                
                return BeautifulSoup(content, 'html.parser')
                
        except Exception as e:
            print(f"❌ Playwright attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(random.uniform(5, 10))
    
    return None

def find_fixtures_table(soup):
    """Find the Premier League fixtures table"""
    print("🔍 Looking for fixtures table...")
    
    # Look for table with fixtures data
    # FBRef typically uses tables with specific IDs or classes
    potential_tables = []
    
    # Check all tables for fixture-like content
    for table in soup.find_all('table'):
        table_id = table.get('id', '')
        table_class = ' '.join(table.get('class', []))
        
        # Look for table headers that indicate fixtures
        headers = []
        header_row = table.find('tr')
        if header_row:
            headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
        
        # Check if this looks like a fixtures table
        fixture_indicators = ['date', 'time', 'home', 'away', 'score', 'result']
        header_text = ' '.join(headers)
        
        if any(indicator in header_text for indicator in fixture_indicators):
            potential_tables.append((table, len(headers), table_id, table_class))
            print(f"✅ Found potential table: ID='{table_id}', Class='{table_class}', Headers={len(headers)}")
    
    # Sort by number of headers (more comprehensive tables first)
    potential_tables.sort(key=lambda x: x[1], reverse=True)
    
    if potential_tables:
        return potential_tables[0][0]
    
    print("⚠️ No fixtures table found")
    return None

def parse_team_name(cell):
    """Extract team name from cell, handling links and nested elements"""
    if not cell:
        return ''
    
    # Try to find the team name in a link first
    link = cell.find('a')
    if link:
        return link.get_text(strip=True)
    
    # Otherwise get the text content
    return cell.get_text(strip=True)

def parse_date(date_str):
    """Parse date string into a standard format"""
    if not date_str:
        return ''
    
    # Clean the date string
    date_str = date_str.strip()
    
    # Handle different date formats
    try:
        # Try common formats
        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%b %d, %Y']:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # If no format works, return as-is
        return date_str
    except:
        return date_str

def verify_correct_season_page(soup, expected_season):
    """Verify that the scraped page contains data for the expected season"""
    
    # Check table ID first - this is the most reliable indicator
    tables = soup.find_all('table')
    expected_season_str = f"{expected_season}-{expected_season+1}"
    
    for table in tables:
        table_id = table.get('id', '')
        if table_id:
            # Look for season in table ID
            if expected_season_str in table_id or str(expected_season) in table_id:
                print(f"✅ Found table with correct season ID: {table_id}")
                return True
            elif 'sched_2024-2025' in table_id and expected_season != 2024:
                print(f"❌ Found 2024-25 table when expecting {expected_season_str}: {table_id}")
                return False
    
    # Check page content for season references
    page_text = soup.get_text()
    
    # Look for actual match data that indicates the season
    # Check for teams that were in Premier League in specific seasons
    if expected_season <= 2016:
        # Teams that were relegated/promoted in early seasons
        early_teams = ['Hull City', 'QPR', 'Burnley']  # Teams that had limited PL years
        found_early_teams = any(team in page_text for team in early_teams)
    
    # Look for current season teams that shouldn't be in historical data
    current_season_teams = ['Ipswich Town', 'Leicester City', 'Southampton']  # 2024-25 promoted/returned teams
    found_current_teams = any(team in page_text for team in current_season_teams)
    
    if expected_season < 2024 and found_current_teams:
        print(f"❌ Found current season teams in historical season {expected_season_str}")
        return False
    
    # Check for season string in page title or headings
    headings = soup.find_all(['h1', 'h2', 'h3', 'title'])
    for heading in headings:
        heading_text = heading.get_text()
        if expected_season_str in heading_text:
            print(f"✅ Found season in heading: {heading_text[:100]}")
            return True
    
    print(f"⚠️ Could not verify season {expected_season_str} - proceeding with caution")
    return True  # Allow it but with warning

def find_fixtures_link(soup, season):
    """Find the fixtures/schedule link on the main season page"""
    season_str = f"{season}-{season+1}"
    
    # Look for schedule/fixtures links
    for link in soup.find_all('a', href=True):
        href = link.get('href', '')
        text = link.text.strip().lower()
        
        # Look for schedule/fixture links
        if ('schedule' in href or 'fixtures' in href) and season_str in href:
            print(f"✅ Found fixtures link: {href}")
            return href
        elif 'scores & fixtures' in text or 'schedule' in text:
            if season_str in href or f"/{season}" in href:
                print(f"✅ Found fixtures link via text: {href}")
                return href
    
    return None

def scrape_season_matches(season):
    """Scrape matches for a specific season"""
    print(f"\n🔍 Scraping matches for season {season}-{season+1}")
    
    fixtures_url = get_fixtures_url(season)
    print(f"📄 Fetching fixtures from: {fixtures_url}")
    
    # Try requests first
    print("🚀 Trying requests method...")
    session = requests.Session()
    soup = scrape_with_requests(fixtures_url, session)
    
    if not soup:
        print("🚀 Trying playwright method...")
        soup = scrape_with_playwright(fixtures_url)
    
    if not soup:
        print("❌ Could not fetch fixtures page")
        return []
    
def scrape_season_matches(season):
    """Scrape matches for a specific season"""
    print(f"\n🔍 Scraping matches for season {season}-{season+1}")
    
    if season == 2024:
        # Current season - go directly to fixtures
        fixtures_url = get_fixtures_url(season)
        print(f"📄 Fetching current season fixtures from: {fixtures_url}")
        
        session = requests.Session()
        soup = scrape_with_requests(fixtures_url, session)
        
        if not soup:
            soup = scrape_with_playwright(fixtures_url)
        
        if not soup:
            print("❌ Could not fetch current season fixtures page")
            return []
            
    else:
        # Historical season - start from main season page, then find fixtures link
        season_url = get_fixtures_url(season)
        print(f"📄 Fetching season page: {season_url}")
        
        session = requests.Session()
        soup = scrape_with_requests(season_url, session)
        
        if not soup:
            soup = scrape_with_playwright(season_url)
        
        if not soup:
            print("❌ Could not fetch season page")
            return []
        
        # Verify we're on the right season page
        if not verify_correct_season_page(soup, season):
            print(f"❌ Skipping season {season} - incorrect page content")
            return []
        
        # Find the fixtures link
        fixtures_link = find_fixtures_link(soup, season)
        if not fixtures_link:
            print("❌ Could not find fixtures link on season page")
            return []
        
        # Navigate to fixtures page
        if fixtures_link.startswith('/'):
            fixtures_url = f"https://fbref.com{fixtures_link}"
        else:
            fixtures_url = fixtures_link
            
        print(f"📄 Fetching fixtures from: {fixtures_url}")
        
        soup = scrape_with_requests(fixtures_url, session)
        if not soup:
            soup = scrape_with_playwright(fixtures_url)
        
        if not soup:
            print("❌ Could not fetch fixtures page")
            return []
    
    # Final verification on fixtures page
    if not verify_correct_season_page(soup, season):
        print(f"❌ Skipping season {season} - fixtures page has incorrect content")
        return []
    
    # Find the fixtures table
    matches_table = find_fixtures_table(soup)
    
    if not matches_table:
        print("⚠️ Could not find fixtures table")
        return []
    
    matches_data = []
    
    # Parse table rows
    rows = matches_table.find_all('tr')
    header_row = rows[0] if rows else None
    
    # Find column indices
    col_indices = {}
    if header_row:
        headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
        print(f"📋 Table headers: {headers}")
        
        # Map common column names to indices
        for i, header in enumerate(headers):
            if 'date' in header:
                col_indices['date'] = i
            elif 'time' in header:
                col_indices['time'] = i
            elif 'home' in header or header == 'team':
                col_indices['home'] = i
            elif 'away' in header or 'visitor' in header:
                col_indices['away'] = i
            elif 'score' in header or 'result' in header:
                col_indices['score'] = i
            elif 'venue' in header:
                col_indices['venue'] = i
    
    print(f"📊 Column mapping: {col_indices}")
    
    # Process data rows
    for row_idx, row in enumerate(rows[1:], 1):  # Skip header row
        cells = row.find_all(['td', 'th'])
        
        if len(cells) < 4:  # Need at least date, home, away, score
            continue
        
        try:
            # Extract data based on column positions or fallback to positional
            if col_indices:
                date = cells[col_indices.get('date', 0)].get_text(strip=True) if col_indices.get('date', 0) < len(cells) else ''
                time_str = cells[col_indices.get('time', 1)].get_text(strip=True) if col_indices.get('time', 1) < len(cells) else ''
                home_team = parse_team_name(cells[col_indices.get('home', 2)]) if col_indices.get('home', 2) < len(cells) else ''
                away_team = parse_team_name(cells[col_indices.get('away', 4)]) if col_indices.get('away', 4) < len(cells) else ''
                score = cells[col_indices.get('score', 3)].get_text(strip=True) if col_indices.get('score', 3) < len(cells) else ''
                venue = cells[col_indices.get('venue', 5)].get_text(strip=True) if col_indices.get('venue', 5) < len(cells) and len(cells) > 5 else ''
            else:
                # Fallback to positional parsing
                date = cells[0].get_text(strip=True) if len(cells) > 0 else ''
                time_str = cells[1].get_text(strip=True) if len(cells) > 1 else ''
                home_team = parse_team_name(cells[2]) if len(cells) > 2 else ''
                score = cells[3].get_text(strip=True) if len(cells) > 3 else ''
                away_team = parse_team_name(cells[4]) if len(cells) > 4 else ''
                venue = cells[5].get_text(strip=True) if len(cells) > 5 else ''
            
            # Skip if no valid team names
            if not home_team or not away_team or len(home_team) < 2 or len(away_team) < 2:
                continue
            
            # Skip header-like rows
            if (home_team.lower() in ['home', 'team', 'date'] or 
                away_team.lower() in ['away', 'team', 'visitor']):
                continue
            
            match_data = {
                'season': f"{season}-{season+1}",
                'date': parse_date(date),
                'time': time_str,
                'home_team': home_team,
                'away_team': away_team,
                'score': score,
                'venue': venue,
                'competition': 'Premier League'
            }
            
            matches_data.append(match_data)
            
            # Debug: show first few matches
            if len(matches_data) <= 3:
                print(f"✅ Match {len(matches_data)}: {home_team} vs {away_team} ({score})")
                
        except Exception as e:
            if row_idx <= 10:  # Only show errors for first few rows
                print(f"⚠️ Error parsing row {row_idx}: {e}")
            continue
    
    print(f"✅ Found {len(matches_data)} Premier League matches for {season}-{season+1}")
    return matches_data

def parse_score(score_text):
    """Parse score text into home and away goals"""
    if not score_text or score_text.strip() == '':
        return pd.NA, pd.NA
    
    # Clean the score text
    score_text = score_text.strip()
    
    # Handle various score formats and separators
    separators = ['–', '−', '-', ':', '‒', '—']
    
    for separator in separators:
        if separator in score_text:
            try:
                parts = score_text.split(separator)
                if len(parts) == 2:
                    # Extract numbers from each part
                    home_part = re.search(r'\d+', parts[0])
                    away_part = re.search(r'\d+', parts[1])
                    
                    if home_part and away_part:
                        home_goals = int(home_part.group())
                        away_goals = int(away_part.group())
                        return home_goals, away_goals
            except (ValueError, AttributeError):
                continue
    
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
    print("🚀 Starting Fixed FBRef Premier League scraper...")
    print(f"📅 Seasons to scrape: {SEASONS}")
    print("💡 Using direct fixtures URLs with improved parsing")
    
    all_matches = []

    for i, season in enumerate(SEASONS):
        print(f"\n🌐 Processing season {season}-{season+1} ({i+1}/{len(SEASONS)})")
        
        if i > 0:
            delay = random.uniform(8, 15)
            print(f"⏳ Waiting {delay:.1f} seconds between seasons...")
            time.sleep(delay)

        try:
            season_matches = scrape_season_matches(season)
            all_matches.extend(season_matches)
            print(f"📈 Completed season {season}-{season+1}, total matches so far: {len(all_matches)}")
        except Exception as e:
            print(f"❌ Error scraping season {season}: {e}")
            continue

    # Process and save data
    if all_matches:
        print(f"\n📊 Processing {len(all_matches)} matches...")
        df = pd.DataFrame(all_matches)

        # Parse scores
        df[['home_goals', 'away_goals']] = df['score'].apply(lambda x: pd.Series(parse_score(x)))
        df['result'] = df.apply(determine_result, axis=1)

        # Clean up data and remove duplicates more thoroughly
        print("🧹 Cleaning data and removing duplicates...")
        
        # First, let's see what we have
        print("Raw data before cleaning:")
        season_counts_raw = df['season'].value_counts().sort_index()
        for season, count in season_counts_raw.items():
            print(f"   {season}: {count} matches")
        
        # Remove exact duplicates
        df = df.drop_duplicates()
        
        # Remove matches that appear to be from wrong seasons
        # (this can happen if URLs redirect to default pages)
        cleaned_df = pd.DataFrame()
        
        for season_str in df['season'].unique():
            season_matches = df[df['season'] == season_str].copy()
            
            # If we have way too many matches for a season, it might be duplicated data
            if len(season_matches) > 500:  # Premier League has ~380 matches per season
                print(f"⚠️ Season {season_str} has {len(season_matches)} matches - likely duplicated")
                # Keep only unique matches based on teams and date
                season_matches = season_matches.drop_duplicates(
                    subset=['date', 'home_team', 'away_team'], 
                    keep='first'
                )
                print(f"   Reduced to {len(season_matches)} matches after deduplication")
            
            cleaned_df = pd.concat([cleaned_df, season_matches], ignore_index=True)
        
        df = cleaned_df.sort_values(['season', 'date']).reset_index(drop=True)
        
        print("Data after cleaning:")
        season_counts_clean = df['season'].value_counts().sort_index()
        for season, count in season_counts_clean.items():
            print(f"   {season}: {count} matches")

        # Save to CSV
        output_path = os.path.join(DATA_DIR, 'fbref_premier_league_matches.csv')
        df.to_csv(output_path, index=False)
        
        print(f"\n💾 Data saved to: {output_path}")
        print(f"✅ Total matches scraped: {len(df)}")
        print(f"📈 Matches with scores: {len(df[~pd.isna(df['home_goals'])])}")
        print(f"🏆 Results distribution:")
        if 'result' in df.columns:
            print(df['result'].value_counts())
        
        # Show season breakdown
        print(f"\n📋 Matches per season:")
        season_counts = df['season'].value_counts().sort_index()
        for season, count in season_counts.items():
            print(f"   {season}: {count} matches")
        
        print(f"\n📋 Sample of scraped data:")
        print(df[['season', 'date', 'home_team', 'away_team', 'score', 'result']].head(5).to_string(index=False))
        
    else:
        print("\n⚠️ No data was collected.")
        print("🔍 Troubleshooting suggestions:")
        print("   1. Check your internet connection")
        print("   2. Try running during off-peak hours")
        print("   3. Consider using a VPN if your IP is being rate-limited")
        print("   4. Test with a single recent season first: SEASONS = [2023]")

if __name__ == "__main__":
    main()