import os
import time
import pandas as pd
import random
import requests
from bs4 import BeautifulSoup, Comment
from playwright.sync_api import sync_playwright
import urllib3
from datetime import datetime
import re
import json
import logging
from urllib.parse import urljoin, urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ========== Setup Paths ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'get_data', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ========== Constants ==========
SEASONS = [2024]  # Example: 2024-25 season
BASE_URL = "https://fbref.com"
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]

MATCH_STATS_MAPPING = {
    'possession': ['possession', 'poss', 'poss%', 'poss %'],
    'shots': ['shots', 'sh', 'total shots', 'total_shots'],
    'shots_on_target': ['shots on target', 'sot', 'on target', 'shots_on_target'],
    'saves': ['saves', 'save', 'sv', 'goalkeeper saves'],
    'corners': ['corner kicks', 'corners', 'corner', 'ck'],
    'fouls': ['fouls', 'foul', 'fl', 'fouls committed'],
    'cards': ['cards', 'card', 'total cards'],
    'yellow_cards': ['yellow cards', 'yellow', 'yc', 'yellow_cards'],
    'red_cards': ['red cards', 'red', 'rc', 'red_cards'],
    'offsides': ['offsides', 'offside', 'off'],
    'passes': ['passes', 'pass', 'total passes', 'pass attempts'],
    'pass_accuracy': ['pass accuracy', 'pass%', 'pass pct', 'pass_accuracy', 'pass %'],
    'crosses': ['crosses', 'cross', 'crs'],
    'touches': ['touches', 'touch', 'tch'],
    'tackles': ['tackles', 'tackle', 'tkl'],
    'interceptions': ['interceptions', 'int', 'interception'],
    'aerials_won': ['aerials won', 'aerial', 'aer', 'aerials'],
    'clearances': ['clearances', 'clear', 'clr'],
    'blocks': ['blocks', 'block', 'blk'],
    'xg': ['xg', 'expected goals', 'exp goals', 'expected_goals'],
    'xa': ['xa', 'expected assists', 'exp assists', 'expected_assists'],
    'big_chances': ['big chances', 'bc', 'big_chances'],
    'big_chances_missed': ['big chances missed', 'bcm', 'big_chances_missed'],
    'distance_covered': ['distance covered', 'distance', 'dist', 'km'],
    'sprints': ['sprints', 'sprint', 'sp'],
    'long_balls': ['long balls', 'long', 'lb'],
    'through_balls': ['through balls', 'through', 'tb'],
    'duels_won': ['duels won', 'duels', 'duel'],
    'errors': ['errors', 'error', 'err']
}

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def get_fixtures_url(season):
    if season == 2024:
        return "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
    else:
        season_str = f"{season}-{season+1}"
        return f"https://fbref.com/en/comps/9/{season_str}/schedule/{season_str}-Premier-League-Scores-and-Fixtures"

def create_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': get_random_user_agent(),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Referer': 'https://fbref.com/',
        'Cache-Control': 'max-age=0'
    })
    return session

def scrape_with_requests(url, session=None, max_retries=3):
    if session is None:
        session = create_session()
    for attempt in range(max_retries):
        try:
            logger.info(f"Requests attempt {attempt + 1}: {url}")
            time.sleep(random.uniform(3, 6))
            response = session.get(url, timeout=30, verify=False)
            response.raise_for_status()
            if response.status_code == 200:
                return BeautifulSoup(response.content, 'html.parser')
            else:
                logger.warning(f"HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Requests attempt {attempt + 1} failed: {e}")
        if attempt < max_retries - 1:
            time.sleep(random.uniform(5, 10))
    return None

def scrape_with_playwright(url, max_retries=2):
    for attempt in range(max_retries):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
                context = browser.new_context(user_agent=get_random_user_agent(), viewport={'width': 1920, 'height': 1080})
                page = context.new_page()
                logger.info(f"Playwright attempt {attempt + 1}: {url}")
                page.goto(url, timeout=60000, wait_until='domcontentloaded')
                try:
                    page.wait_for_selector('table', timeout=15000)
                except:
                    logger.warning("No tables found initially")
                    time.sleep(random.uniform(5, 10))
                content = page.content()
                browser.close()
                return BeautifulSoup(content, 'html.parser')
        except Exception as e:
            logger.error(f"Playwright attempt {attempt + 1} failed: {e}")
        if attempt < max_retries - 1:
            time.sleep(random.uniform(5, 10))
    return None

def get_all_tables_including_comments(soup):
    tables = list(soup.find_all('table'))
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        try:
            cs = BeautifulSoup(comment, 'html.parser')
            tables.extend(cs.find_all('table'))
        except Exception:
            pass
    return tables

def extract_match_stats_comprehensive(soup, home_team, away_team):
    """Comprehensive extraction that handles FBRef's actual structure"""
    stats = {'home_stats': {}, 'away_stats': {}}
    
    # 1. Extract from team stats comparison table - FIXED POSSESSION EXTRACTION
    team_stats_table = soup.find('table', id='team_stats')
    if team_stats_table:
        rows = team_stats_table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) < 3:
                continue
                
            stat_name = cells[0].get_text(strip=True).lower()
            home_val = cells[1].get_text(strip=True)
            away_val = cells[2].get_text(strip=True)
            
            # Special handling for possession
            if 'possession' in stat_name:
                # Extract numeric value from percentage strings
                home_match = re.search(r'(\d+)%', home_val)
                away_match = re.search(r'(\d+)%', away_val)
                home_val = home_match.group(1) if home_match else None
                away_val = away_match.group(1) if away_match else None
            
            mapped_stat = map_stat_name(stat_name)
            if mapped_stat:
                if home_val and home_val not in ['', '-', '—']:
                    stats['home_stats'][mapped_stat] = home_val
                if away_val and away_val not in ['', '-', '—']:
                    stats['away_stats'][mapped_stat] = away_val
    
    # 2. Extract from supplemental stats section
    supplemental_stats = soup.find('div', class_='supplemental_stats')
    if supplemental_stats:
        for row in supplemental_stats.find_all('div', class_='row'):
            stat_name = row.find('div', class_='stat').get_text(strip=True).lower()
            home_val = row.find('div', class_='home_value').get_text(strip=True)
            away_val = row.find('div', class_='away_value').get_text(strip=True)
            
            mapped_stat = map_stat_name(stat_name)
            if mapped_stat:
                if home_val and home_val not in ['', '-', '—']:
                    stats['home_stats'][mapped_stat] = home_val
                if away_val and away_val not in ['', '-', '—']:
                    stats['away_stats'][mapped_stat] = away_val
    
    # 3. Extract from summary tables
    for table in soup.find_all('table', id=re.compile(r'stats_\w+_summary$')):
        caption = table.find('caption')
        if not caption:
            continue
            
        caption_text = caption.get_text(strip=True).lower()
        is_home = home_team.lower() in caption_text
        is_away = away_team.lower() in caption_text
        
        if not is_home and not is_away:
            continue
            
        team_key = 'home_stats' if is_home else 'away_stats'
        
        # Extract from table footer
        tfoot = table.find('tfoot')
        if tfoot:
            footer_row = tfoot.find('tr')
            if footer_row:
                # Get headers
                thead = table.find('thead')
                if thead:
                    header_row = thead.find_all('tr')[-1]
                    headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
                    
                    # Get footer values
                    cells = footer_row.find_all(['td', 'th'])
                    
                    for i, cell in enumerate(cells):
                        if i >= len(headers):
                            continue
                            
                        stat_name = headers[i]
                        value = cell.get_text(strip=True)
                        
                        # Special handling for percentage stats
                        if stat_name.endswith('%') and value.endswith('%'):
                            value = value.replace('%', '')
                        
                        mapped_stat = map_stat_name(stat_name)
                        if mapped_stat and value and value not in ['', '-', '—']:
                            stats[team_key][mapped_stat] = value
    
    # 4. Extract from shooting tables
    for table in soup.find_all('table', id=re.compile(r'stats_\w+_shooting$')):
        caption = table.find('caption')
        if not caption:
            continue
            
        caption_text = caption.get_text(strip=True).lower()
        is_home = home_team.lower() in caption_text
        is_away = away_team.lower() in caption_text
        
        if not is_home and not is_away:
            continue
            
        team_key = 'home_stats' if is_home else 'away_stats'
        
        # Look for xG row
        for row in table.find_all('tr'):
            cells = row.find_all(['td', 'th'])
            if len(cells) < 2:
                continue
                
            if 'xg' in cells[0].get_text().lower():
                xg_val = cells[-1].get_text(strip=True)
                if xg_val and xg_val not in ['', '-', '—']:
                    stats[team_key]['xg'] = xg_val
                break
    
    # 5. Extract goalkeeper stats for saves
    for table in soup.find_all('table', id=re.compile(r'keeper_stats')):
        caption = table.find('caption')
        if not caption:
            continue
            
        caption_text = caption.get_text(strip=True).lower()
        is_home = home_team.lower() in caption_text
        is_away = away_team.lower() in caption_text
        
        if not is_home and not is_away:
            continue
            
        team_key = 'home_stats' if is_home else 'away_stats'
        
        # Get saves
        tbody = table.find('tbody')
        if tbody:
            for row in tbody.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) > 3:
                    saves_cell = cells[3]
                    if saves_cell and saves_cell.get_text(strip=True) not in ['', '-', '—']:
                        stats[team_key]['saves'] = saves_cell.get_text(strip=True)
                        break
    
    # 6. FALLBACK: Extract possession from scorebox if still missing
    if 'possession' not in stats['home_stats']:
        scorebox = soup.find('div', class_='scorebox')
        if scorebox:
            poss_divs = scorebox.find_all('div', class_=re.compile('scorebox_meta-value'))
            for div in poss_divs:
                if 'possession' in div.parent.get_text().lower():
                    parts = div.get_text(strip=True).split('—')
                    if len(parts) == 2:
                        home_val = re.sub(r'\D', '', parts[0])  # Extract digits only
                        away_val = re.sub(r'\D', '', parts[1])
                        stats['home_stats']['possession'] = home_val
                        stats['away_stats']['possession'] = away_val
    
    logger.info(f"Extracted stats - Home: {len(stats['home_stats'])} stats, Away: {len(stats['away_stats'])} stats")
    return stats

def map_stat_name(stat_name):
    stat_name_lower = stat_name.lower()
    for mapped_name, variations in MATCH_STATS_MAPPING.items():
        if any(variation in stat_name_lower for variation in variations):
            return mapped_name
    return None

def find_fixtures_table(soup):
    logger.info("Looking for fixtures table...")
    table_identifiers = ['sched', 'fixture', 'schedule', 'scores']
    for table in soup.find_all('table'):
        table_id = table.get('id', '').lower()
        table_class = ' '.join(table.get('class', [])).lower()
        if any(identifier in table_id for identifier in table_identifiers):
            logger.info(f"Found fixtures table with ID: {table_id}")
            return table
        if any(identifier in table_class for identifier in table_identifiers):
            logger.info(f"Found fixtures table with class: {table_class}")
            return table
    for table in soup.find_all('table'):
        rows = table.find_all('tr')
        if len(rows) > 1:
            first_row = rows[0]
            cells = first_row.find_all(['th', 'td'])
            if len(cells) >= 5:
                logger.info("Using first table with sufficient columns")
                return table
    return None

def extract_team_name(cell):
    if not cell:
        return ''
    link = cell.find('a')
    if link:
        return link.get_text(strip=True)
    return cell.get_text(strip=True)

def parse_date(date_str):
    if not date_str:
        return ''
    date_str = date_str.strip()
    date_formats = [
        '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%b %d, %Y',
        '%B %d, %Y', '%d %b %Y', '%d %B %Y'
    ]
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            return parsed_date.strftime('%Y-%m-%d')
        except ValueError:
            continue
    return date_str

def parse_score(score_str):
    """Parse score string into home and away goals"""
    if not score_str:
        return None, None
    score_str = score_str.replace('−', '-').replace('–', '-').replace(':', '-').replace('‒', '-').replace('—', '-')
    score_parts = re.findall(r'\d+', score_str)
    if len(score_parts) >= 2:
        try:
            home_goals = int(score_parts[0])
            away_goals = int(score_parts[1])
            return home_goals, away_goals
        except:
            return None, None
    return None, None

def determine_result(match):
    """Determine match result based on score"""
    if not match.get('score'):
        return None
    home_goals, away_goals = parse_score(match['score'])
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return 'Home Win'
    elif away_goals > home_goals:
        return 'Away Win'
    else:
        return 'Draw'

def get_match_report_url(match_link):
    if not match_link:
        return None
    if match_link.startswith('http'):
        return match_link
    elif match_link.startswith('/'):
        return urljoin(BASE_URL, match_link)
    else:
        return urljoin(BASE_URL, '/' + match_link)

def scrape_season_fixtures(season, session=None):
    logger.info(f"Scraping fixtures for season {season}-{season+1}")
    if session is None:
        session = create_session()
    fixtures_url = get_fixtures_url(season)
    logger.info(f"Fetching fixtures from: {fixtures_url}")
    soup = scrape_with_requests(fixtures_url, session)
    if not soup:
        soup = scrape_with_playwright(fixtures_url)
    if not soup:
        logger.error("Could not fetch fixtures page")
        return []
    fixtures_table = find_fixtures_table(soup)
    if not fixtures_table:
        logger.error("Could not find fixtures table")
        return []
    matches = []
    rows = fixtures_table.find_all('tr')
    if not rows:
        logger.error("No rows found in fixtures table")
        return []
    header_row = rows[0]
    headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
    logger.info(f"Table headers: {headers}")
    col_map = {}
    for i, header in enumerate(headers):
        if 'date' in header:
            col_map['date'] = i
        elif 'time' in header:
            col_map['time'] = i
        elif 'home' in header or header in ['team', 'squad']:
            col_map['home'] = i
        elif 'away' in header or 'visitor' in header:
            col_map['away'] = i
        elif 'score' in header or 'result' in header:
            col_map['score'] = i
        elif 'venue' in header:
            col_map['venue'] = i
        elif 'report' in header or 'match report' in header:
            col_map['report'] = i
    for row_idx, row in enumerate(rows[1:], 1):
        cells = row.find_all(['td', 'th'])
        if len(cells) < 4:
            continue
        try:
            match_data = {
                'season': f"{season}-{season+1}",
                'date': parse_date(cells[col_map.get('date', 0)].get_text(strip=True)) if col_map.get('date', 0) < len(cells) else '',
                'time': cells[col_map.get('time', 1)].get_text(strip=True) if col_map.get('time', 1) < len(cells) else '',
                'home_team': extract_team_name(cells[col_map.get('home', 2)]) if col_map.get('home', 2) < len(cells) else '',
                'away_team': extract_team_name(cells[col_map.get('away', 4)]) if col_map.get('away', 4) < len(cells) else '',
                'score': cells[col_map.get('score', 3)].get_text(strip=True) if col_map.get('score', 3) < len(cells) else '',
                'venue': cells[col_map.get('venue', 5)].get_text(strip=True) if col_map.get('venue', 5) < len(cells) else '',
                'match_report_url': None
            }
            if 'report' in col_map and col_map['report'] < len(cells):
                report_cell = cells[col_map['report']]
                report_link = report_cell.find('a', href=True)
                if report_link:
                    match_data['match_report_url'] = get_match_report_url(report_link['href'])
            else:
                for cell in cells:
                    link = cell.find('a', href=True)
                    if link and '/matches/' in link.get('href', ''):
                        match_data['match_report_url'] = get_match_report_url(link['href'])
                        break
            if (not match_data['home_team'] or not match_data['away_team'] or
                len(match_data['home_team']) < 2 or len(match_data['away_team']) < 2):
                continue
            if (match_data['home_team'].lower() in ['home', 'team', 'date'] or
                match_data['away_team'].lower() in ['away', 'team', 'visitor']):
                continue
            matches.append(match_data)
            if len(matches) <= 3:
                logger.info(f"Match {len(matches)}: {match_data['home_team']} vs {match_data['away_team']} ({match_data['score']})")
        except Exception as e:
            if row_idx <= 10:
                logger.warning(f"Error parsing row {row_idx}: {e}")
            continue
    logger.info(f"Found {len(matches)} matches for season {season}-{season+1}")
    return matches

def enhance_match_with_stats(match, session=None):
    """Enhance match with comprehensive stats"""
    if not match.get('match_report_url'):
        return match
    
    if session is None:
        session = create_session()
    
    try:
        logger.info(f"Fetching stats for: {match['home_team']} vs {match['away_team']}")
        logger.info(f"URL: {match['match_report_url']}")
        
        time.sleep(random.uniform(3, 7))
        
        # Try with requests first
        soup = scrape_with_requests(match['match_report_url'], session)
        
        if not soup:
            logger.info("Requests failed, trying Playwright...")
            soup = scrape_with_playwright(match['match_report_url'])
        
        if not soup:
            logger.warning(f"Could not fetch match report for {match['home_team']} vs {match['away_team']}")
            return match
        
        # Extract stats using comprehensive method
        stats = extract_match_stats_comprehensive(soup, match['home_team'], match['away_team'])
        
        # Add stats to match data
        for stat_name in MATCH_STATS_MAPPING.keys():
            home_key = f"home_{stat_name}"
            away_key = f"away_{stat_name}"
            
            home_value = stats['home_stats'].get(stat_name)
            away_value = stats['away_stats'].get(stat_name)
            
            match[home_key] = home_value if home_value else None
            match[away_key] = away_value if away_value else None
        
        logger.info(f"Enhanced with stats for {match['home_team']} vs {match['away_team']}")
        
    except Exception as e:
        logger.error(f"Error enhancing match stats: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return match

def main():
    logger.info("Starting Enhanced FBRef Premier League scraper...")
    logger.info(f"Seasons to scrape: {SEASONS}")
    all_matches = []
    session = create_session()

    # Step 1: Scrape all fixtures
    for i, season in enumerate(SEASONS):
        logger.info(f"Processing season {season}-{season+1} ({i+1}/{len(SEASONS)})")
        if i > 0:
            delay = random.uniform(5, 15)
            logger.info(f"Waiting {delay:.1f} seconds between seasons...")
            time.sleep(delay)
        try:
            season_matches = scrape_season_fixtures(season, session)
            all_matches.extend(season_matches)
            logger.info(f"Completed season {season}-{season+1}, total matches: {len(all_matches)}")
        except Exception as e:
            logger.error(f"Error scraping season {season}: {e}")
            continue
    if not all_matches:
        logger.error("No matches were collected")
        return

    # Step 2: Save basic fixtures data
    logger.info(f"Saving basic fixtures data for {len(all_matches)} matches...")
    df_basic = pd.DataFrame(all_matches)
    df_basic[['home_goals', 'away_goals']] = df_basic['score'].apply(lambda x: pd.Series(parse_score(x)))
    df_basic['result'] = df_basic.apply(determine_result, axis=1)
    basic_path = os.path.join(DATA_DIR, 'fbref_premier_league_basic.csv')
    df_basic.to_csv(basic_path, index=False)
    logger.info(f"Basic fixtures saved to: {basic_path}")

    # Step 3: Enhance matches with detailed stats
    logger.info("Enhancing matches with detailed statistics...")
    matches_to_enhance = [m for m in all_matches if m.get('match_report_url') and m.get('score') and any(sep in m.get('score', '') for sep in ['–', '−', '-', ':', '‒', '—'])]
    logger.info(f"Enhancing {len(matches_to_enhance)} matches with detailed stats...")
    enhanced_matches = []
    for i, match in enumerate(matches_to_enhance):
        logger.info(f"Enhancing match {i+1}/{len(matches_to_enhance)}: {match['home_team']} vs {match['away_team']}")
        enhanced_match = enhance_match_with_stats(match, session)
        enhanced_matches.append(enhanced_match)
    
    # Step 4: Save enhanced data
    enhanced_path = os.path.join(DATA_DIR, 'fbref_premier_league_enhanced.csv')
    df_enhanced = pd.DataFrame(enhanced_matches)
    
    # Ensure all columns exist in the DataFrame
    expected_columns = [
        'season', 'date', 'time', 'home_team', 'away_team', 'score', 'venue', 'match_report_url',
        'home_possession', 'away_possession', 'home_shots', 'away_shots', 
        'home_shots_on_target', 'away_shots_on_target', 'home_saves', 'away_saves',
        'home_corners', 'away_corners', 'home_fouls', 'away_fouls', 'home_cards', 'away_cards',
        'home_yellow_cards', 'away_yellow_cards', 'home_red_cards', 'away_red_cards',
        'home_offsides', 'away_offsides', 'home_passes', 'away_passes', 
        'home_pass_accuracy', 'away_pass_accuracy', 'home_crosses', 'away_crosses',
        'home_touches', 'away_touches', 'home_tackles', 'away_tackles',
        'home_interceptions', 'away_interceptions', 'home_aerials_won', 'away_aerials_won',
        'home_clearances', 'away_clearances', 'home_blocks', 'away_blocks',
        'home_xg', 'away_xg', 'home_xa', 'away_xa', 'home_big_chances', 'away_big_chances',
        'home_big_chances_missed', 'away_big_chances_missed', 'home_distance_covered', 
        'away_distance_covered', 'home_sprints', 'away_sprints', 'home_long_balls', 
        'away_long_balls', 'home_through_balls', 'away_through_balls', 
        'home_duels_won', 'away_duels_won', 'home_errors', 'away_errors'
    ]
    
    # Add missing columns with None values
    for col in expected_columns:
        if col not in df_enhanced.columns:
            df_enhanced[col] = None
    
    # Reorder columns to match expected format
    df_enhanced = df_enhanced[expected_columns]
    df_enhanced.to_csv(enhanced_path, index=False)
    logger.info(f"Enhanced fixtures saved to: {enhanced_path}")
    
    # Step 5: Print sample output
    if len(df_enhanced) > 0:
        sample = df_enhanced.iloc[0].to_dict()
        logger.info("Sample enhanced match data:")
        for k, v in sample.items():
            logger.info(f"{k}: {v}")

if __name__ == "__main__":
    main()