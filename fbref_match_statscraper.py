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
import json

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

# Enhanced metrics mapping for different table types
MATCH_STATS_MAPPING = {
    'shots': ['shots', 'sh', 'total_shots'],
    'shots_on_target': ['sot', 'shots_on_target', 'on_target'],
    'possession': ['poss', 'possession', 'poss%'],
    'passes': ['passes', 'pass', 'total_passes'],
    'pass_accuracy': ['pass%', 'pass_accuracy', 'pass_pct'],
    'corners': ['corners', 'corner', 'ck'],
    'fouls': ['fouls', 'foul', 'fl'],
    'yellow_cards': ['yellow', 'yc', 'yellow_cards'],
    'red_cards': ['red', 'rc', 'red_cards'],
    'offsides': ['offsides', 'off', 'offside'],
    'saves': ['saves', 'save', 'sv'],
    'crosses': ['crosses', 'cross', 'crs'],
    'tackles': ['tackles', 'tackle', 'tkl'],
    'interceptions': ['int', 'interceptions', 'interception'],
    'clearances': ['clr', 'clearances', 'clear'],
    'aerials_won': ['aerials', 'aerial', 'aer'],
    'blocks': ['blocks', 'block', 'blk'],
    'errors': ['errors', 'error', 'err'],
    'big_chances': ['big_chances', 'bc', 'big_chance'],
    'big_chances_missed': ['big_chances_missed', 'bcm', 'bc_missed'],
    'xg': ['xg', 'expected_goals', 'exp_goals'],
    'xa': ['xa', 'expected_assists', 'exp_assists'],
    'distance_covered': ['distance', 'dist', 'km'],
    'sprints': ['sprints', 'sprint', 'sp'],
    'duels_won': ['duels', 'duel', 'duel_won'],
    'long_balls': ['long_balls', 'long', 'lb'],
    'through_balls': ['through_balls', 'through', 'tb']
}

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def get_fixtures_url(season):
    """Get the direct fixtures URL for a specific season"""
    if season == 2024:  # Current season 2024-25
        return "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
    else:
        # Historical seasons
        season_str = f"{season}-{season+1}"
        return f"https://fbref.com/en/comps/9/{season_str}/schedule/{season_str}-Premier-League-Scores-and-Fixtures"

def get_match_report_url(match_link):
    """Convert relative match link to full URL"""
    if match_link.startswith('/'):
        return f"https://fbref.com{match_link}"
    return match_link

def scrape_with_requests(url, session=None, max_retries=3):
    """Enhanced scraping method using requests"""
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
    """Enhanced Playwright scraping"""
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
                
                page.goto(url, timeout=60000, wait_until='domcontentloaded')
                
                # Wait for content to load
                try:
                    page.wait_for_selector('table', timeout=15000)
                except:
                    print("⚠️ No tables found initially")
                
                # Additional wait for dynamic content
                time.sleep(random.uniform(3, 8))
                
                content = page.content()
                browser.close()
                
                return BeautifulSoup(content, 'html.parser')
                
        except Exception as e:
            print(f"❌ Playwright attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(random.uniform(5, 10))
    
    return None

def extract_match_links(soup):
    """Extract match report links from fixtures page"""
    match_links = []
    
    # Look for match report links
    for link in soup.find_all('a', href=True):
        href = link.get('href', '')
        if '/matches/' in href and 'match-report' in href:
            match_links.append(href)
    
    return list(set(match_links))  # Remove duplicates

def find_fixtures_table(soup):
    """Find the Premier League fixtures table"""
    print("🔍 Looking for fixtures table...")
    
    potential_tables = []
    
    for table in soup.find_all('table'):
        table_id = table.get('id', '')
        table_class = ' '.join(table.get('class', []))
        
        # Look for fixtures table indicators
        if 'sched' in table_id or 'fixture' in table_id or 'schedule' in table_id:
            potential_tables.append(table)
            print(f"✅ Found fixtures table: ID='{table_id}', Class='{table_class}'")
    
    return potential_tables[0] if potential_tables else None

def extract_detailed_match_stats(soup, home_team, away_team):
    """Extract detailed match statistics from match report page"""
    stats = {
        'home_stats': {},
        'away_stats': {}
    }
    
    # Look for stats tables
    for table in soup.find_all('table'):
        table_id = table.get('id', '')
        
        # Skip if not a stats table
        if not any(keyword in table_id for keyword in ['stats', 'summary', 'keeper', 'passing', 'defense']):
            continue
            
        rows = table.find_all('tr')
        if len(rows) < 3:  # Need header + 2 team rows
            continue
            
        # Find header row
        header_row = None
        for row in rows:
            cells = row.find_all(['th', 'td'])
            if len(cells) >= 3:
                header_row = row
                break
        
        if not header_row:
            continue
            
        headers = [cell.get_text(strip=True).lower() for cell in header_row.find_all(['th', 'td'])]
        
        # Find team rows
        team_rows = []
        for row in rows[1:]:  # Skip header
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 3:
                team_name = cells[0].get_text(strip=True) if cells else ''
                if team_name and len(team_name) > 2:  # Valid team name
                    team_rows.append((team_name, cells))
        
        if len(team_rows) != 2:
            continue
            
        # Map stats to teams
        for team_name, cells in team_rows:
            # Determine if this is home or away team
            is_home = any(ht in team_name for ht in home_team.split()) or team_name in home_team
            is_away = any(at in team_name for at in away_team.split()) or team_name in away_team
            
            if not (is_home or is_away):
                continue
                
            team_key = 'home_stats' if is_home else 'away_stats'
            
            # Extract stats based on headers
            for i, header in enumerate(headers):
                if i < len(cells):
                    value = cells[i].get_text(strip=True)
                    
                    # Map header to standardized stat name
                    stat_name = None
                    for std_name, variations in MATCH_STATS_MAPPING.items():
                        if any(var in header for var in variations):
                            stat_name = std_name
                            break
                    
                    if stat_name and value:
                        # Clean and convert numeric values
                        clean_value = re.sub(r'[^\d.,%-]', '', value)
                        if clean_value:
                            stats[team_key][stat_name] = clean_value
    
    return stats

def parse_team_name(cell):
    """Extract team name from cell"""
    if not cell:
        return ''
    
    link = cell.find('a')
    if link:
        return link.get_text(strip=True)
    
    return cell.get_text(strip=True)

def parse_date(date_str):
    """Parse date string into standard format"""
    if not date_str:
        return ''
    
    date_str = date_str.strip()
    
    try:
        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%b %d, %Y']:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        return date_str
    except:
        return date_str

def scrape_enhanced_season_matches(season, max_detailed_matches=50):
    """Scrape matches with enhanced statistics for a specific season"""
    print(f"\n🔍 Scraping enhanced matches for season {season}-{season+1}")
    
    fixtures_url = get_fixtures_url(season)
    print(f"📄 Fetching fixtures from: {fixtures_url}")
    
    # Get fixtures page
    session = requests.Session()
    soup = scrape_with_requests(fixtures_url, session)
    
    if not soup:
        soup = scrape_with_playwright(fixtures_url)
    
    if not soup:
        print("❌ Could not fetch fixtures page")
        return []
    
    # Extract match links for detailed stats
    match_links = extract_match_links(soup)
    print(f"🔗 Found {len(match_links)} match report links")
    
    # Find fixtures table
    matches_table = find_fixtures_table(soup)
    if not matches_table:
        print("⚠️ Could not find fixtures table")
        return []
    
    matches_data = []
    
    # Parse basic match data
    rows = matches_table.find_all('tr')
    header_row = rows[0] if rows else None
    
    # Find column indices
    col_indices = {}
    if header_row:
        headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
        print(f"📋 Table headers: {headers}")
        
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
            elif 'report' in header or 'match report' in header:
                col_indices['report'] = i
    
    # Process data rows
    detailed_count = 0
    for row_idx, row in enumerate(rows[1:], 1):
        cells = row.find_all(['td', 'th'])
        
        if len(cells) < 4:
            continue
        
        try:
            # Extract basic match data
            if col_indices:
                date = cells[col_indices.get('date', 0)].get_text(strip=True) if col_indices.get('date', 0) < len(cells) else ''
                time_str = cells[col_indices.get('time', 1)].get_text(strip=True) if col_indices.get('time', 1) < len(cells) else ''
                home_team = parse_team_name(cells[col_indices.get('home', 2)]) if col_indices.get('home', 2) < len(cells) else ''
                away_team = parse_team_name(cells[col_indices.get('away', 4)]) if col_indices.get('away', 4) < len(cells) else ''
                score = cells[col_indices.get('score', 3)].get_text(strip=True) if col_indices.get('score', 3) < len(cells) else ''
                venue = cells[col_indices.get('venue', 5)].get_text(strip=True) if col_indices.get('venue', 5) < len(cells) and len(cells) > 5 else ''
            else:
                date = cells[0].get_text(strip=True) if len(cells) > 0 else ''
                time_str = cells[1].get_text(strip=True) if len(cells) > 1 else ''
                home_team = parse_team_name(cells[2]) if len(cells) > 2 else ''
                score = cells[3].get_text(strip=True) if len(cells) > 3 else ''
                away_team = parse_team_name(cells[4]) if len(cells) > 4 else ''
                venue = cells[5].get_text(strip=True) if len(cells) > 5 else ''
            
            if not home_team or not away_team or len(home_team) < 2 or len(away_team) < 2:
                continue
            
            if (home_team.lower() in ['home', 'team', 'date'] or 
                away_team.lower() in ['away', 'team', 'visitor']):
                continue
            
            # Initialize match data with basic info
            match_data = {
                'season': f"{season}-{season+1}",
                'date': parse_date(date),
                'time': time_str,
                'home_team': home_team,
                'away_team': away_team,
                'score': score,
                'venue': venue,
                'competition': 'Premier League',
                # Initialize enhanced stats
                'home_shots': None,
                'away_shots': None,
                'home_shots_on_target': None,
                'away_shots_on_target': None,
                'home_possession': None,
                'away_possession': None,
                'home_passes': None,
                'away_passes': None,
                'home_pass_accuracy': None,
                'away_pass_accuracy': None,
                'home_corners': None,
                'away_corners': None,
                'home_fouls': None,
                'away_fouls': None,
                'home_yellow_cards': None,
                'away_yellow_cards': None,
                'home_red_cards': None,
                'away_red_cards': None,
                'home_offsides': None,
                'away_offsides': None,
                'home_xg': None,
                'away_xg': None,
                'home_xa': None,
                'away_xa': None
            }
            
            # Try to get detailed stats if we haven't reached limit
            if detailed_count < max_detailed_matches and score and '–' in score:
                # Look for match report link in this row
                match_link = None
                for cell in cells:
                    link = cell.find('a', href=True)
                    if link and '/matches/' in link.get('href', ''):
                        match_link = link.get('href')
                        break
                
                if match_link:
                    try:
                        match_url = get_match_report_url(match_link)
                        print(f"📊 Fetching detailed stats for: {home_team} vs {away_team}")
                        
                        # Add delay before fetching detailed stats
                        time.sleep(random.uniform(3, 8))
                        
                        match_soup = scrape_with_requests(match_url, session)
                        if match_soup:
                            detailed_stats = extract_detailed_match_stats(match_soup, home_team, away_team)
                            
                            # Update match data with detailed stats
                            for stat_name in match_data.keys():
                                if stat_name.startswith('home_'):
                                    base_stat = stat_name.replace('home_', '')
                                    if base_stat in detailed_stats['home_stats']:
                                        match_data[stat_name] = detailed_stats['home_stats'][base_stat]
                                elif stat_name.startswith('away_'):
                                    base_stat = stat_name.replace('away_', '')
                                    if base_stat in detailed_stats['away_stats']:
                                        match_data[stat_name] = detailed_stats['away_stats'][base_stat]
                            
                            detailed_count += 1
                            print(f"✅ Got detailed stats ({detailed_count}/{max_detailed_matches})")
                        
                    except Exception as e:
                        print(f"⚠️ Could not fetch detailed stats: {e}")
            
            matches_data.append(match_data)
            
            if len(matches_data) <= 3:
                print(f"✅ Match {len(matches_data)}: {home_team} vs {away_team} ({score})")
                
        except Exception as e:
            if row_idx <= 10:
                print(f"⚠️ Error parsing row {row_idx}: {e}")
            continue
    
    print(f"✅ Found {len(matches_data)} matches ({detailed_count} with detailed stats) for {season}-{season+1}")
    return matches_data

def parse_score(score_text):
    """Parse score text into home and away goals"""
    if not score_text or score_text.strip() == '':
        return pd.NA, pd.NA
    
    score_text = score_text.strip()
    separators = ['–', '−', '-', ':', '‒', '—']
    
    for separator in separators:
        if separator in score_text:
            try:
                parts = score_text.split(separator)
                if len(parts) == 2:
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

def clean_numeric_stat(value):
    """Clean and convert stat values to numeric"""
    if pd.isna(value) or value is None:
        return pd.NA
    
    # Convert to string and clean
    value = str(value)
    
    # Remove percentage signs and convert percentages
    if '%' in value:
        try:
            return float(value.replace('%', '')) / 100
        except:
            return pd.NA
    
    # Clean general numeric values
    cleaned = re.sub(r'[^\d.,]', '', value)
    if cleaned:
        try:
            return float(cleaned.replace(',', ''))
        except:
            return pd.NA
    
    return pd.NA

def main():
    print("🚀 Starting Enhanced FBRef Premier League scraper...")
    print(f"📅 Seasons to scrape: {SEASONS}")
    print("📊 Collecting detailed match statistics for better predictions")
    
    all_matches = []
    
    for i, season in enumerate(SEASONS):
        print(f"\n🌐 Processing season {season}-{season+1} ({i+1}/{len(SEASONS)})")
        
        if i > 0:
            delay = random.uniform(10, 20)
            print(f"⏳ Waiting {delay:.1f} seconds between seasons...")
            time.sleep(delay)
        
        try:
            # Limit detailed stats collection for older seasons to avoid overwhelming the site
            max_detailed = 30 if season < 2022 else 50
            season_matches = scrape_enhanced_season_matches(season, max_detailed_matches=max_detailed)
            all_matches.extend(season_matches)
            print(f"📈 Completed season {season}-{season+1}, total matches: {len(all_matches)}")
        except Exception as e:
            print(f"❌ Error scraping season {season}: {e}")
            continue
    
    # Process and save enhanced data
    if all_matches:
        print(f"\n📊 Processing {len(all_matches)} matches with enhanced metrics...")
        df = pd.DataFrame(all_matches)
        
        # Parse scores
        df[['home_goals', 'away_goals']] = df['score'].apply(lambda x: pd.Series(parse_score(x)))
        df['result'] = df.apply(determine_result, axis=1)
        
        # Clean numeric stats
        numeric_stats = [col for col in df.columns if any(stat in col for stat in 
                        ['shots', 'possession', 'passes', 'corners', 'fouls', 'cards', 'offsides', 'xg', 'xa'])]
        
        for stat in numeric_stats:
            df[stat] = df[stat].apply(clean_numeric_stat)
        
        # Calculate derived metrics
        df['home_shot_accuracy'] = df['home_shots_on_target'] / df['home_shots'].where(df['home_shots'] > 0)
        df['away_shot_accuracy'] = df['away_shots_on_target'] / df['away_shots'].where(df['away_shots'] > 0)
        
        # Goal difference
        df['goal_difference'] = df['home_goals'] - df['away_goals']
        
        # Clean and deduplicate
        df = df.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='first')
        df = df.sort_values(['season', 'date']).reset_index(drop=True)
        
        # Save enhanced data
        output_path = os.path.join(DATA_DIR, 'fbref_premier_league_enhanced.csv')
        df.to_csv(output_path, index=False)
        
        print(f"\n💾 Enhanced data saved to: {output_path}")
        print(f"✅ Total matches: {len(df)}")
        print(f"📊 Matches with detailed stats: {len(df[df['home_shots'].notna()])}")
        
        # Show available metrics
        print(f"\n📋 Available metrics for prediction:")
        metrics = [col for col in df.columns if any(stat in col for stat in 
                  ['shots', 'possession', 'passes', 'corners', 'fouls', 'cards', 'xg', 'xa'])]
        for metric in sorted(metrics):
            non_null = df[metric].notna().sum()
            print(f"   {metric}: {non_null} matches")
        
        # Sample of enhanced data
        print(f"\n📋 Sample enhanced data:")
        sample_cols = ['season', 'home_team', 'away_team', 'score', 'home_shots', 'away_shots', 
                      'home_possession', 'away_possession', 'home_xg', 'away_xg']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].head(3).to_string(index=False))
        
    else:
        print("\n⚠️ No data was collected.")

if __name__ == "__main__":
    main()