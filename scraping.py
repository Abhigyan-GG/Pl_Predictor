# Updated PLScraper class with fixed team scraping
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PLScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.base_url = "https://www.premierleague.com"
        self.teams = {}
        self.matches = []
        self.team_stats = {}
        
    def get_teams(self):
        """Scrape the current Premier League teams - UPDATED for new website structure"""
        url = f"{self.base_url}/clubs"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code != 200:
            logging.error(f"Failed to retrieve teams: Status code {response.status_code}")
            return self.teams
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Debug information
        logging.info(f"Retrieved clubs page with status {response.status_code}")
        
        # Try multiple possible selectors to find team elements
        selectors = [
            '.indexItem',  # Original selector
            '.team-card', 
            '.club-card',
            'li.clubList__club',
            '.club'
        ]
        
        team_elements = []
        
        for selector in selectors:
            team_elements = soup.select(selector)
            if team_elements:
                logging.info(f"Found {len(team_elements)} teams using selector: {selector}")
                break
                
        if not team_elements:
            # If no teams found with predefined selectors, let's try to find links containing '/clubs/'
            logging.info("Using fallback method to find team links")
            all_links = soup.find_all('a', href=True)
            team_links = [link for link in all_links if '/clubs/' in link['href'] and 'profile' not in link['href']]
            
            # Deduplicate
            unique_urls = set()
            unique_team_links = []
            
            for link in team_links:
                url = link['href']
                if url not in unique_urls:
                    unique_urls.add(url)
                    unique_team_links.append(link)
            
            logging.info(f"Found {len(unique_team_links)} unique team links via fallback method")
            
            # Get team names from these links
            for link in unique_team_links:
                # Extract team name from URL or try to find it in the link text
                team_url = link['href']
                team_id = team_url.split('/')[-1]
                
                # Try to get the team name from the link text or an img alt if present
                team_name = link.text.strip()
                if not team_name:
                    img = link.find('img')
                    if img and img.has_attr('alt'):
                        team_name = img['alt'].replace(' Club Profile', '').strip()
                
                # If still no name, use the ID from the URL
                if not team_name:
                    team_name = team_id.replace('-', ' ').title()
                
                self.teams[team_name] = {
                    'url': self.base_url + team_url if not team_url.startswith('http') else team_url,
                    'id': team_id
                }
            
            return self.teams
        
        # If we found team elements with a selector
        for team in team_elements:
            # Try multiple ways to find the team name
            team_name = None
            name_selectors = ['.clubName', '.team-card__name', '.club-name', '.name', 'h2', 'h3']
            
            for name_selector in name_selectors:
                name_element = team.select_one(name_selector)
                if name_element:
                    team_name = name_element.text.strip()
                    break
            
            # Try to find team link
            link_element = None
            if team.name == 'a':
                link_element = team
            else:
                link_element = team.find('a')
            
            team_url = None
            team_id = None
            
            if link_element and link_element.has_attr('href'):
                team_url = link_element['href']
                team_id = team_url.split('/')[-1]
            
            # If we still don't have a name, try img alt
            if not team_name:
                img = team.find('img')
                if img and img.has_attr('alt'):
                    team_name = img['alt'].replace(' Club Profile', '').strip()
            
            # If we found both name and URL
            if team_name and team_url:
                self.teams[team_name] = {
                    'url': self.base_url + team_url if not team_url.startswith('http') else team_url,
                    'id': team_id
                }
        
        if not self.teams:
            logging.error("Could not extract any teams!")
            # As a last resort, hardcode the current Premier League teams
            self.teams = self._get_hardcoded_teams()
            
        return self.teams
    
    def _get_hardcoded_teams(self):
        """Fallback method with hardcoded team data"""
        logging.info("Using hardcoded team data as fallback")
        teams = {
            "Arsenal": {"id": "arsenal", "url": "https://www.premierleague.com/clubs/1/Arsenal/overview"},
            "Aston Villa": {"id": "aston-villa", "url": "https://www.premierleague.com/clubs/2/Aston-Villa/overview"},
            "Brentford": {"id": "brentford", "url": "https://www.premierleague.com/clubs/130/Brentford/overview"},
            "Brighton": {"id": "brighton-and-hove-albion", "url": "https://www.premierleague.com/clubs/131/Brighton-and-Hove-Albion/overview"},
            "Bournemouth": {"id": "afc-bournemouth", "url": "https://www.premierleague.com/clubs/127/AFC-Bournemouth/overview"},
            "Chelsea": {"id": "chelsea", "url": "https://www.premierleague.com/clubs/4/Chelsea/overview"},
            "Crystal Palace": {"id": "crystal-palace", "url": "https://www.premierleague.com/clubs/6/Crystal-Palace/overview"},
            "Everton": {"id": "everton", "url": "https://www.premierleague.com/clubs/7/Everton/overview"},
            "Fulham": {"id": "fulham", "url": "https://www.premierleague.com/clubs/34/Fulham/overview"},
            "Leicester City": {"id": "leicester-city", "url": "https://www.premierleague.com/clubs/26/Leicester-City/overview"},
            "Liverpool": {"id": "liverpool", "url": "https://www.premierleague.com/clubs/10/Liverpool/overview"},
            "Manchester City": {"id": "manchester-city", "url": "https://www.premierleague.com/clubs/11/Manchester-City/overview"},
            "Manchester United": {"id": "manchester-united", "url": "https://www.premierleague.com/clubs/12/Manchester-United/overview"},
            "Newcastle United": {"id": "newcastle-united", "url": "https://www.premierleague.com/clubs/23/Newcastle-United/overview"},
            "Nottingham Forest": {"id": "nottingham-forest", "url": "https://www.premierleague.com/clubs/15/Nottingham-Forest/overview"},
            "Southampton": {"id": "southampton", "url": "https://www.premierleague.com/clubs/20/Southampton/overview"},
            "Tottenham Hotspur": {"id": "tottenham-hotspur", "url": "https://www.premierleague.com/clubs/21/Tottenham-Hotspur/overview"},
            "West Ham United": {"id": "west-ham-united", "url": "https://www.premierleague.com/clubs/25/West-Ham-United/overview"},
            "Wolverhampton": {"id": "wolverhampton-wanderers", "url": "https://www.premierleague.com/clubs/38/Wolverhampton-Wanderers/overview"},
            "Ipswich Town": {"id": "ipswich-town", "url": "https://www.premierleague.com/clubs/633/Ipswich-Town/overview"}
        }
        return teams

    def get_fixtures(self, season="2024-25", status="all"):
        """Scrape fixtures - status can be all, LIVE, U (upcoming), or R (results)"""
        url = f"{self.base_url}/fixtures"
        params = {
            'season': season,
            'status': status
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            logging.info(f"Retrieved fixtures page with status {response.status_code}")
            
            if response.status_code != 200:
                logging.error(f"Failed to retrieve fixtures: Status code {response.status_code}")
                return []
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            fixtures = []
            fixture_elements = soup.select('.fixtures__matches-list .matchFixtureContainer')
            
            if not fixture_elements:
                logging.info("Original fixture selector not found, trying alternative selectors")
                fixture_selectors = [
                    '.fixture',
                    '.match-fixture',
                    '.matchWeek__fixture',
                    '.match-card'
                ]
                
                for selector in fixture_selectors:
                    fixture_elements = soup.select(selector)
                    if fixture_elements:
                        logging.info(f"Found {len(fixture_elements)} fixtures using selector: {selector}")
                        break
            
            if not fixture_elements:
                logging.error("Could not find any fixtures with known selectors")
                return []
                
            for fixture in fixture_elements:
                try:
                    # Try to find date in various ways
                    match_date_element = fixture.find_previous('time')
                    if not match_date_element:
                        match_date_element = fixture.select_one('time')
                    
                    match_date = None
                    if match_date_element and match_date_element.has_attr('datetime'):
                        match_date = match_date_element['datetime']
                    elif match_date_element:
                        match_date = match_date_element.text.strip()
                    
                    # Try different team selectors
                    team_selectors = [
                        ('.team.home .js-team', '.team.away .js-team'),
                        ('.home-team', '.away-team'),
                        ('.team--home', '.team--away')
                    ]
                    
                    home_team = away_team = None
                    
                    for home_sel, away_sel in team_selectors:
                        home_elem = fixture.select_one(home_sel)
                        away_elem = fixture.select_one(away_sel)
                        
                        if home_elem and away_elem:
                            if home_elem.has_attr('data-short'):
                                home_team = home_elem['data-short']
                            else:
                                home_team = home_elem.text.strip()
                                
                            if away_elem.has_attr('data-short'):
                                away_team = away_elem['data-short']
                            else:
                                away_team = away_elem.text.strip()
                            
                            break
                    
                    if not home_team or not away_team:
                        continue
                    
                    # Try to find score
                    score_element = fixture.select_one('.score')
                    if not score_element:
                        score_selectors = ['.score', '.scoreline', '.fixture__score', '.match-score']
                        for selector in score_selectors:
                            score_element = fixture.select_one(selector)
                            if score_element:
                                break
                    
                    home_score = away_score = None
                    status = 'upcoming'
                    
                    if score_element and '-' in score_element.text:
                        home_score, away_score = score_element.text.strip().split('-')
                        status = 'completed'
                    
                    fixtures.append({
                        'date': match_date,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score,
                        'status': status
                    })
                    
                except Exception as e:
                    logging.error(f"Error parsing fixture: {str(e)}")
                    continue
            
            logging.info(f"Successfully scraped {len(fixtures)} fixtures")
            return fixtures
            
        except Exception as e:
            logging.error(f"Error scraping fixtures: {str(e)}")
            return []

    def get_team_stats(self, team_id):
        """Get detailed stats for a specific team"""
        url = f"{self.base_url}/clubs/{team_id}/stats"
        try:
            response = requests.get(url, headers=self.headers)
            logging.info(f"Retrieved team stats page for {team_id} with status {response.status_code}")
            
            if response.status_code != 200:
                logging.error(f"Failed to retrieve team stats: Status code {response.status_code}")
                return {}
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            stats = {}
            stat_blocks = soup.select('.statCard')
            
            if not stat_blocks:
                logging.info("Original stat block selector not found, trying alternatives")
                stat_selectors = ['.statCard', '.statsCard', '.stat-card', '.stats-card']
                
                for selector in stat_selectors:
                    stat_blocks = soup.select(selector)
                    if stat_blocks:
                        logging.info(f"Found {len(stat_blocks)} stat blocks using selector: {selector}")
                        break
            
            if not stat_blocks:
                logging.error(f"Could not find any stat blocks for {team_id}")
                return {}
            
            for block in stat_blocks:
                try:
                    # Try different title selectors
                    title_selectors = ['.statCardTitle', '.card-title', 'h3', 'h4']
                    category = None
                    
                    for selector in title_selectors:
                        title_elem = block.select_one(selector)
                        if title_elem:
                            category = title_elem.text.strip()
                            break
                    
                    if not category:
                        continue
                        
                    stats[category] = {}
                    
                    # Try different stat item selectors
                    stat_items = block.select('.statCardContent .statistic')
                    if not stat_items:
                        stat_selectors = [
                            '.statCardContent .statistic', 
                            '.stat-item',
                            '.stat',
                            '.statistic'
                        ]
                        
                        for selector in stat_selectors:
                            stat_items = block.select(selector)
                            if stat_items:
                                break
                    
                    for item in stat_items:
                        try:
                            # Try different name/value selectors
                            name_selectors = ['.caption', '.stat-name', '.label']
                            value_selectors = ['.value', '.stat-value', '.data']
                            
                            stat_name = None
                            for selector in name_selectors:
                                name_elem = item.select_one(selector)
                                if name_elem:
                                    stat_name = name_elem.text.strip()
                                    break
                            
                            stat_value = None
                            for selector in value_selectors:
                                value_elem = item.select_one(selector)
                                if value_elem:
                                    stat_value = value_elem.text.strip()
                                    break
                            
                            if stat_name and stat_value:
                                stats[category][stat_name] = stat_value
                                
                        except Exception as e:
                            logging.error(f"Error parsing stat item: {str(e)}")
                            continue
                            
                except Exception as e:
                    logging.error(f"Error parsing stat block: {str(e)}")
                    continue
            
            return stats
            
        except Exception as e:
            logging.error(f"Error scraping team stats: {str(e)}")
            return {}

    # Other methods remain the same
    def get_historical_results(self, team_id, num_seasons=2):
        """Get historical match results for a team"""
        results = []
        for season in range(2024 - num_seasons, 2025):
            season_str = f"{season}-{str(season+1)[-2:]}"
            url = f"{self.base_url}/clubs/{team_id}/results"
            params = {'season': season_str}
            
            try:
                response = requests.get(url, headers=self.headers, params=params)
                logging.info(f"Retrieved results for {team_id} ({season_str}) with status {response.status_code}")
                
                if response.status_code != 200:
                    logging.error(f"Failed to retrieve results: Status code {response.status_code}")
                    continue
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                
                match_elements = soup.select('.matchFixtureContainer')
                if not match_elements:
                    logging.info("Original match selector not found, trying alternatives")
                    match_selectors = [
                        '.fixture',
                        '.match-fixture',
                        '.matchWeek__fixture',
                        '.match-card'
                    ]
                    
                    for selector in match_selectors:
                        match_elements = soup.select(selector)
                        if match_elements:
                            logging.info(f"Found {len(match_elements)} matches using selector: {selector}")
                            break
                
                if not match_elements:
                    logging.error(f"Could not find any match results for {team_id}")
                    continue
                
                for match in match_elements:
                    try:
                        # Find date
                        date_element = match.find_previous('time')
                        if not date_element:
                            date_element = match.select_one('time')
                        
                        match_date = None
                        if date_element and date_element.has_attr('datetime'):
                            match_date = date_element['datetime']
                        elif date_element:
                            match_date = date_element.text.strip()
                        
                        # Find teams
                        team_selectors = [
                            ('.team.home .js-team', '.team.away .js-team'),
                            ('.home-team', '.away-team'),
                            ('.team--home', '.team--away')
                        ]
                        
                        home_team = away_team = None
                        
                        for home_sel, away_sel in team_selectors:
                            home_elem = match.select_one(home_sel)
                            away_elem = match.select_one(away_sel)
                            
                            if home_elem and away_elem:
                                if home_elem.has_attr('data-short'):
                                    home_team = home_elem['data-short']
                                else:
                                    home_team = home_elem.text.strip()
                                    
                                if away_elem.has_attr('data-short'):
                                    away_team = away_elem['data-short']
                                else:
                                    away_team = away_elem.text.strip()
                                
                                break
                        
                        if not home_team or not away_team:
                            continue
                        
                        # Find score
                        score_element = match.select_one('.score')
                        if not score_element:
                            score_selectors = ['.score', '.scoreline', '.fixture__score', '.match-score']
                            for selector in score_selectors:
                                score_element = match.select_one(selector)
                                if score_element:
                                    break
                        
                        if not score_element or '-' not in score_element.text:
                            continue  # Skip if no score (match not completed)
                        
                        # Parse scores
                        score_text = score_element.text.strip()
                        home_score, away_score = map(lambda s: int(s.strip()), score_text.split('-'))
                        
                        result = {
                            'date': match_date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_score': home_score,
                            'away_score': away_score,
                            'season': season_str
                        }
                        
                        # Determine match outcome from perspective of the team we're looking at
                        if home_team == team_id:
                            if home_score > away_score:
                                result['outcome'] = 'W'
                            elif home_score < away_score:
                                result['outcome'] = 'L'
                            else:
                                result['outcome'] = 'D'
                        else:
                            if away_score > home_score:
                                result['outcome'] = 'W'
                            elif away_score < home_score:
                                result['outcome'] = 'L'
                            else:
                                result['outcome'] = 'D'
                        
                        results.append(result)
                        
                    except Exception as e:
                        logging.error(f"Error parsing match: {str(e)}")
                        continue
            
            except Exception as e:
                logging.error(f"Error scraping historical results: {str(e)}")
                continue
        
        logging.info(f"Successfully scraped {len(results)} historical results for {team_id}")
        return results

    def get_table(self):
        """Get current Premier League table"""
        url = f"{self.base_url}/tables"
        try:
            response = requests.get(url, headers=self.headers)
            logging.info(f"Retrieved table page with status {response.status_code}")
            
            if response.status_code != 200:
                logging.error(f"Failed to retrieve table: Status code {response.status_code}")
                return pd.DataFrame()
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            table_rows = soup.select('.tableBodyContainer tr')
            if not table_rows:
                logging.info("Original table row selector not found, trying alternatives")
                row_selectors = [
                    '.tableBodyContainer tr',
                    '.league-table tr',
                    '.table-body tr',
                    '.standings-table tr'
                ]
                
                for selector in row_selectors:
                    table_rows = soup.select(selector)
                    if table_rows:
                        logging.info(f"Found {len(table_rows)} table rows using selector: {selector}")
                        break
            
            if not table_rows:
                logging.error("Could not find any table rows")
                return pd.DataFrame()
                
            table_data = []
            
            for row in table_rows:
                try:
                    # Try to find position
                    position_elem = row.select_one('.value')
                    if not position_elem:
                        position_selectors = ['.value', '.position', '.pos', '.rank']
                        for selector in position_selectors:
                            position_elem = row.select_one(selector)
                            if position_elem:
                                break
                    
                    if not position_elem:
                        continue
                        
                    position = position_elem.text.strip()
                    
                    # Try to find team name
                    team_elem = row.select_one('.clubName')
                    if not team_elem:
                        team_selectors = ['.clubName', '.team-name', '.name', '.club']
                        for selector in team_selectors:
                            team_elem = row.select_one(selector)
                            if team_elem:
                                break
                    
                    if not team_elem:
                        continue
                        
                    team = team_elem.text.strip()
                    
                    # Find columns
                    columns = row.select('.resultHighlight')
                    if not columns or len(columns) < 8:
                        column_selectors = ['.resultHighlight', '.stats', '.stat', 'td']
                        for selector in column_selectors:
                            columns = row.select(selector)
                            if columns and len(columns) >= 8:
                                break
                    
                    if not columns or len(columns) < 8:
                        logging.error(f"Not enough columns found for team {team}")
                        continue
                    
                    # Parse the stats
                    played = columns[0].text.strip()
                    won = columns[1].text.strip()
                    drawn = columns[2].text.strip()
                    lost = columns[3].text.strip()
                    gf = columns[4].text.strip()
                    ga = columns[5].text.strip()
                    gd = columns[6].text.strip()
                    points = columns[7].text.strip()
                    
                    # Basic validation
                    if not all(x.strip().isdigit() for x in [played, won, drawn, lost, gf, ga, points]):
                        if not gd[0] in ['+', '-'] and not gd[1:].isdigit():
                            logging.warning(f"Non-numeric stats found for {team}: {played}, {won}, {drawn}, {lost}, {gf}, {ga}, {gd}, {points}")
                            continue
                    
                    table_data.append({
                        'position': int(position),
                        'team': team,
                        'played': int(played),
                        'won': int(won),
                        'drawn': int(drawn),
                        'lost': int(lost),
                        'gf': int(gf),
                        'ga': int(ga),
                        'gd': int(gd.replace('+', '')),  # Remove + sign if present
                        'points': int(points)
                    })
                    
                except Exception as e:
                    logging.error(f"Error parsing table row: {str(e)}")
                    continue
            
            logging.info(f"Successfully scraped table with {len(table_data)} teams")
            return pd.DataFrame(table_data)
            
        except Exception as e:
            logging.error(f"Error scraping table: {str(e)}")
            return pd.DataFrame()
        
    def scrape_all_data(self):
        """Run a complete scraping session to gather all needed data"""
        logging.info("Getting teams...")
        teams = self.get_teams()
        
        if not teams:
            logging.error("Failed to scrape teams, using hardcoded data")
            teams = self._get_hardcoded_teams()
            self.teams = teams
        
        logging.info(f"Found {len(teams)} teams")
        
        try:
            logging.info("Getting fixtures...")
            fixtures = self.get_fixtures()
            logging.info(f"Found {len(fixtures)} fixtures")
            
            logging.info("Getting results...")
            results = []
            for team_name, team_data in self.teams.items():
                logging.info(f"Getting results for {team_name}...")
                team_results = self.get_historical_results(team_data['id'])
                results.extend(team_results)
                time.sleep(1)  # Be nice to the server
            
            logging.info(f"Found {len(results)} total historical results")
            
            logging.info("Getting team stats...")
            team_stats = {}
            for team_name, team_data in self.teams.items():
                logging.info(f"Getting stats for {team_name}...")
                team_stats[team_name] = self.get_team_stats(team_data['id'])
                time.sleep(1)  # Be nice to the server
            
            logging.info("Getting league table...")
            table = self.get_table()
            
            # Save all data
            data = {
                'teams': self.teams,
                'fixtures': fixtures,
                'results': results,
                'team_stats': team_stats,
                'table': [] if table.empty else table.to_dict('records')
            }
            
            with open('pl_data.pkl', 'wb') as f:
                pickle.dump(data, f)
            
            logging.info("Data successfully scraped and saved to pl_data.pkl")
            return data
            
        except Exception as e:
            logging.error(f"Error during scraping process: {str(e)}")
            
            # Try to save whatever data we have
            data = {
                'teams': self.teams,
                'fixtures': getattr(self, 'fixtures', []),
                'results': getattr(self, 'results', []),
                'team_stats': getattr(self, 'team_stats', {}),
                'table': []
            }
            
            with open('pl_data.pkl', 'wb') as f:
                pickle.dump(data, f)
            
            logging.info("Partial data saved to pl_data.pkl")
            return data

# Example usage
if __name__ == "__main__":
    scraper = PLScraper()
    data = scraper.scrape_all_data()
    print(f"Scraped data for {len(data['teams'])} teams")