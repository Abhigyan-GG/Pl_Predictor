# Premier League Predictor
# A complete solution for scraping, processing, modeling and displaying PL match predictions

# ---- 1. WEB SCRAPING ----
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

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
        """Scrape the current Premier League teams"""
        url = f"{self.base_url}/clubs"
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        team_elements = soup.select('.indexItem')
        
        for team in team_elements:
            team_name = team.select_one('.clubName').text.strip()
            team_url = team.select_one('a')['href']
            self.teams[team_name] = {
                'url': self.base_url + team_url,
                'id': team_url.split('/')[-1]
            }
        
        return self.teams
    
    def get_fixtures(self, season="2024-25", status="all"):
        """Scrape fixtures - status can be all, LIVE, U (upcoming), or R (results)"""
        url = f"{self.base_url}/fixtures"
        params = {
            'season': season,
            'status': status
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        fixtures = []
        fixture_elements = soup.select('.fixtures__matches-list .matchFixtureContainer')
        
        for fixture in fixture_elements:
            match_date_element = fixture.find_previous('time')
            match_date = match_date_element['datetime'] if match_date_element else None
            
            home_team = fixture.select_one('.team.home .js-team')['data-short']
            away_team = fixture.select_one('.team.away .js-team')['data-short']
            
            score_element = fixture.select_one('.score')
            if score_element and '-' in score_element.text:
                home_score, away_score = score_element.text.strip().split('-')
                status = 'completed'
            else:
                home_score = away_score = None
                status = 'upcoming'
                
            fixtures.append({
                'date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'status': status
            })
        
        return fixtures
    
    def get_team_stats(self, team_id):
        """Get detailed stats for a specific team"""
        url = f"{self.base_url}/clubs/{team_id}/stats"
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        stats = {}
        stat_blocks = soup.select('.statCard')
        
        for block in stat_blocks:
            category = block.select_one('.statCardTitle').text.strip()
            stats[category] = {}
            
            stat_items = block.select('.statCardContent .statistic')
            for item in stat_items:
                stat_name = item.select_one('.caption').text.strip()
                stat_value = item.select_one('.value').text.strip()
                stats[category][stat_name] = stat_value
        
        return stats
    
    def get_historical_results(self, team_id, num_seasons=2):
        """Get historical match results for a team"""
        results = []
        for season in range(2024 - num_seasons, 2025):
            season_str = f"{season}-{str(season+1)[-2:]}"
            url = f"{self.base_url}/clubs/{team_id}/results"
            params = {'season': season_str}
            
            response = requests.get(url, headers=self.headers, params=params)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            match_elements = soup.select('.matchFixtureContainer')
            for match in match_elements:
                try:
                    date_element = match.find_previous('time')
                    match_date = date_element['datetime'] if date_element else None
                    
                    home_team = match.select_one('.team.home .js-team')['data-short']
                    away_team = match.select_one('.team.away .js-team')['data-short']
                    
                    score_element = match.select_one('.score')
                    if score_element and '-' in score_element.text:
                        home_score, away_score = map(int, score_element.text.strip().split('-'))
                    else:
                        continue  # Skip if no score (match not completed)
                    
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
                    print(f"Error parsing match: {e}")
                    continue
        
        return results

    def get_table(self):
        """Get current Premier League table"""
        url = f"{self.base_url}/tables"
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        table_rows = soup.select('.tableBodyContainer tr')
        table_data = []
        
        for row in table_rows:
            position = row.select_one('.value').text.strip()
            team = row.select_one('.clubName').text.strip()
            
            columns = row.select('.resultHighlight')
            played = columns[0].text.strip()
            won = columns[1].text.strip()
            drawn = columns[2].text.strip()
            lost = columns[3].text.strip()
            gf = columns[4].text.strip()
            ga = columns[5].text.strip()
            gd = columns[6].text.strip()
            points = columns[7].text.strip()
            
            table_data.append({
                'position': int(position),
                'team': team,
                'played': int(played),
                'won': int(won),
                'drawn': int(drawn),
                'lost': int(lost),
                'gf': int(gf),
                'ga': int(ga),
                'gd': int(gd),
                'points': int(points)
            })
        
        return pd.DataFrame(table_data)
        
    def scrape_all_data(self):
        """Run a complete scraping session to gather all needed data"""
        print("Getting teams...")
        self.get_teams()
        
        print("Getting fixtures...")
        fixtures = self.get_fixtures()
        
        print("Getting results...")
        results = []
        for team_name, team_data in self.teams.items():
            print(f"Getting results for {team_name}...")
            team_results = self.get_historical_results(team_data['id'])
            results.extend(team_results)
            time.sleep(1)  # Be nice to the server
        
        print("Getting team stats...")
        team_stats = {}
        for team_name, team_data in self.teams.items():
            print(f"Getting stats for {team_name}...")
            team_stats[team_name] = self.get_team_stats(team_data['id'])
            time.sleep(1)  # Be nice to the server
        
        print("Getting league table...")
        table = self.get_table()
        
        # Save all data
        data = {
            'teams': self.teams,
            'fixtures': fixtures,
            'results': results,
            'team_stats': team_stats,
            'table': table.to_dict('records')
        }
        
        with open('pl_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        return data

# ---- 2. DATA PROCESSING ----
class PLDataProcessor:
    def __init__(self, data_path='pl_data.pkl'):
        self.load_data(data_path)
        
    def load_data(self, path):
        """Load data from pickle file"""
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
            
        self.teams = self.data['teams']
        self.fixtures = pd.DataFrame(self.data['fixtures'])
        self.results = pd.DataFrame(self.data['results'])
        self.team_stats = self.data['team_stats']
        self.table = pd.DataFrame(self.data['table'])
    
    def prepare_feature_set(self):
        """Prepare features for the prediction model"""
        # Convert results to DataFrame if not already
        if isinstance(self.results, list):
            results_df = pd.DataFrame(self.results)
        else:
            results_df = self.results.copy()
        
        # Ensure date is datetime format
        results_df['date'] = pd.to_datetime(results_df['date'])
        results_df = results_df.sort_values('date')
        
        # Create dataframes for home and away performances
        home_df = results_df[['date', 'home_team', 'away_team', 'home_score', 'away_score']]
        home_df['team'] = home_df['home_team']  
        home_df['opponent'] = home_df['away_team']
        home_df['goals_scored'] = home_df['home_score']
        home_df['goals_conceded'] = home_df['away_score']
        home_df['is_home'] = 1
                
        away_df = results_df[['date', 'home_team', 'away_team', 'home_score', 'away_score']]
        away_df['team'] = away_df['away_team']
        away_df['opponent'] = away_df['home_team']
        away_df['goals_scored'] = away_df['away_score']
        away_df['goals_conceded'] = away_df['home_score']
        away_df['is_home'] = 0
        
        team_matches = pd.concat([home_df, away_df], ignore_index=True)
        team_matches = team_matches.sort_values(['team', 'date'])
        
        # Calculate results (win=1, draw=0.5, loss=0)
        team_matches['result'] = (team_matches['goals_scored'] > team_matches['goals_conceded']).astype(float)
        team_matches.loc[team_matches['goals_scored'] == team_matches['goals_conceded'], 'result'] = 0.5
        
        # Calculate rolling averages for team performance metrics
        features = ['result', 'goals_scored', 'goals_conceded', 'is_home']
        windows = [3, 5, 10]
        
        for feature in features:
            for window in windows:
                team_matches[f'{feature}_last_{window}'] = team_matches.groupby('team')[feature].transform(
                    lambda x: x.shift().rolling(window=window, min_periods=1).mean()
                )
        
        # Create head-to-head features
        h2h = pd.DataFrame()
        for home_team in team_matches['team'].unique():
            for away_team in team_matches['team'].unique():
                if home_team != away_team:
                    # Previous matches between these two teams
                    prev_matches = team_matches[
                        ((team_matches['team'] == home_team) & (team_matches['opponent'] == away_team)) |
                        ((team_matches['team'] == away_team) & (team_matches['opponent'] == home_team))
                    ].sort_values('date')
                    
                    if len(prev_matches) > 0:
                        last_match = prev_matches.iloc[-1]
                        
                        h2h_record = {
                            'home_team': home_team,
                            'away_team': away_team,
                            'last_match_date': last_match['date'],
                            'home_wins': sum((prev_matches['team'] == home_team) & (prev_matches['result'] == 1)),
                            'away_wins': sum((prev_matches['team'] == away_team) & (prev_matches['result'] == 1)),
                            'draws': sum(prev_matches['result'] == 0.5),
                            'matches_count': len(prev_matches)
                        }
                        
                        h2h = pd.concat([h2h, pd.DataFrame([h2h_record])], ignore_index=True)
        
        # Merge table stats for current strength indicators
        table_df = pd.DataFrame(self.data['table'])
        team_stats = {}
        
        for _, row in table_df.iterrows():
            team_name = row['team']
            team_stats[team_name] = {
                'position': row['position'],
                'points': row['points'],
                'gd': row['gd'],
                'win_rate': row['won'] / row['played'] if row['played'] > 0 else 0
            }
        
        # Now prepare the feature set for upcoming matches
        upcoming_fixtures = self.fixtures[self.fixtures['status'] == 'upcoming'].copy()
        
        X_features = []
        match_details = []
        
        for _, fixture in upcoming_fixtures.iterrows():
            home_team = fixture['home_team']
            away_team = fixture['away_team']
            match_date = pd.to_datetime(fixture['date'])
            
            # Get latest stats for home team
            home_team_matches = team_matches[team_matches['team'] == home_team].sort_values('date')
            if len(home_team_matches) >= 5:
                home_recent = home_team_matches.iloc[-5:]
                home_form = home_recent['result'].mean()
                home_goals_scored_avg = home_recent['goals_scored'].mean()
                home_goals_conceded_avg = home_recent['goals_conceded'].mean()
            else:
                home_form = 0.5  # Default if not enough matches
                home_goals_scored_avg = 1.0
                home_goals_conceded_avg = 1.0
            
            # Get latest stats for away team
            away_team_matches = team_matches[team_matches['team'] == away_team].sort_values('date')
            if len(away_team_matches) >= 5:
                away_recent = away_team_matches.iloc[-5:]
                away_form = away_recent['result'].mean()
                away_goals_scored_avg = away_recent['goals_scored'].mean()
                away_goals_conceded_avg = away_recent['goals_conceded'].mean()
            else:
                away_form = 0.5  # Default if not enough matches
                away_goals_scored_avg = 1.0
                away_goals_conceded_avg = 1.0
            
            # Get head-to-head stats
            h2h_row = h2h[(h2h['home_team'] == home_team) & (h2h['away_team'] == away_team)]
            if len(h2h_row) > 0:
                h2h_record = h2h_row.iloc[0]
                h2h_home_win_rate = h2h_record['home_wins'] / h2h_record['matches_count'] if h2h_record['matches_count'] > 0 else 0.5
                h2h_matches_count = h2h_record['matches_count']
            else:
                h2h_home_win_rate = 0.5
                h2h_matches_count = 0
            
            # Get current table positions
            try:
                home_position = team_stats[home_team]['position']
                away_position = team_stats[away_team]['position']
                position_diff = away_position - home_position  # Positive if home team is higher ranked
            except:
                home_position = 10
                away_position = 10
                position_diff = 0
            
            # Build feature vector
            features = [
                home_form, away_form,
                home_goals_scored_avg, home_goals_conceded_avg,
                away_goals_scored_avg, away_goals_conceded_avg,
                h2h_home_win_rate, h2h_matches_count,
                home_position, away_position, position_diff
            ]
            
            X_features.append(features)
            match_details.append({
                'home_team': home_team,
                'away_team': away_team,
                'date': match_date
            })
        
        # Convert to array
        self.X_upcoming = np.array(X_features)
        self.upcoming_matches = match_details
        
        # Also prepare training dataset from historical matches
        X_train = []
        y_train = []
        
        # We need to recreate similar features for historical matches
        for idx, match in results_df.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            match_date = pd.to_datetime(match['date'])
            
            # Only use matches with enough prior data
            home_prev_matches = team_matches[
                (team_matches['team'] == home_team) & 
                (team_matches['date'] < match_date)
            ].sort_values('date')
            
            away_prev_matches = team_matches[
                (team_matches['team'] == away_team) & 
                (team_matches['date'] < match_date)
            ].sort_values('date')
            
            if len(home_prev_matches) < 3 or len(away_prev_matches) < 3:
                continue
            
            # Get recent form (last 5 matches)
            home_recent = home_prev_matches.iloc[-5:] if len(home_prev_matches) >= 5 else home_prev_matches
            home_form = home_recent['result'].mean()
            home_goals_scored_avg = home_recent['goals_scored'].mean()
            home_goals_conceded_avg = home_recent['goals_conceded'].mean()
            
            away_recent = away_prev_matches.iloc[-5:] if len(away_prev_matches) >= 5 else away_prev_matches
            away_form = away_recent['result'].mean()
            away_goals_scored_avg = away_recent['goals_scored'].mean()
            away_goals_conceded_avg = away_recent['goals_conceded'].mean()
            
            # Get head-to-head stats prior to this match
            h2h_matches = results_df[
                ((results_df['home_team'] == home_team) & (results_df['away_team'] == away_team) |
                 (results_df['home_team'] == away_team) & (results_df['away_team'] == home_team)) &
                (results_df['date'] < match_date)
            ]
            
            h2h_home_wins = sum((h2h_matches['home_team'] == home_team) & (h2h_matches['home_score'] > h2h_matches['away_score']))
            h2h_away_wins = sum((h2h_matches['home_team'] == away_team) & (h2h_matches['home_score'] > h2h_matches['away_score']))
            h2h_draws = sum(h2h_matches['home_score'] == h2h_matches['away_score'])
            h2h_matches_count = len(h2h_matches)
            h2h_home_win_rate = h2h_home_wins / h2h_matches_count if h2h_matches_count > 0 else 0.5
            
            # Use approximate league positions from recent matches
            home_position = 10  # Default mid-table
            away_position = 10
            position_diff = 0
            
            features = [
                home_form, away_form,
                home_goals_scored_avg, home_goals_conceded_avg,
                away_goals_scored_avg, away_goals_conceded_avg,
                h2h_home_win_rate, h2h_matches_count,
                home_position, away_position, position_diff
            ]
            
            X_train.append(features)
            
            # Create target variable (0 = away win, 1 = draw, 2 = home win)
            if match['home_score'] > match['away_score']:
                result = 2  # Home win
            elif match['home_score'] == match['away_score']:
                result = 1  # Draw
            else:
                result = 0  # Away win
                
            y_train.append(result)
        
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        
        # Define feature names for interpretability
        self.feature_names = [
            'home_form', 'away_form',
            'home_goals_scored_avg', 'home_goals_conceded_avg',
            'away_goals_scored_avg', 'away_goals_conceded_avg',
            'h2h_home_win_rate', 'h2h_matches_count',
            'home_position', 'away_position', 'position_diff'
        ]
        
        return self.X_train, self.y_train, self.X_upcoming, self.upcoming_matches

# ---- 3. PREDICTION MODEL ----
class PLPredictor:
    def __init__(self, data_processor=None):
        self.model = None
        self.scaler = StandardScaler()
        self.data_processor = data_processor
        self.feature_importances = None
    
    def train_model(self):
        """Train the prediction model"""
        if self.data_processor is None:
            raise ValueError("Data processor must be initialized first")
        
        X_train, y_train, _, _ = self.data_processor.prepare_feature_set()
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train a Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Validation accuracy: {accuracy:.4f}")
        print(classification_report(y_val, y_pred, target_names=['Away Win', 'Draw', 'Home Win']))
        
        # Store feature importances
        self.feature_importances = pd.DataFrame({
            'feature': self.data_processor.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return accuracy
    
    def predict_matches(self):
        """Make predictions for upcoming matches"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        _, _, X_upcoming, upcoming_matches = self.data_processor.prepare_feature_set()
        
        # Scale the features
        X_upcoming_scaled = self.scaler.transform(X_upcoming)
        
        # Get predicted probabilities
        probabilities = self.model.predict_proba(X_upcoming_scaled)
        
        # Prepare results
        predictions = []
        for i, match in enumerate(upcoming_matches):
            probs = probabilities[i]
            
            prediction = {
                'home_team': match['home_team'],
                'away_team': match['away_team'],
                'date': match['date'],
                'away_win_prob': probs[0],
                'draw_prob': probs[1],
                'home_win_prob': probs[2],
                'prediction': ['Away Win', 'Draw', 'Home Win'][np.argmax(probs)]
            }
            
            predictions.append(prediction)
        
        self.predictions = pd.DataFrame(predictions)
        return self.predictions
    
    def save_model(self, filepath='pl_predictor_model.pkl'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No trained model to save")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_importances': self.feature_importances
            }, f)
            
    def load_model(self, filepath='pl_predictor_model.pkl'):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_importances = data['feature_importances']

# ---- 4. WEB INTERFACE ----
class PLPredictorApp:
    def __init__(self):
        pass
    
    def run(self):
        st.title("Premier League Match Predictor")
        
        # Sidebar for options
        st.sidebar.title("Options")
        action = st.sidebar.radio("Select Action", 
                                ["View Predictions", "View Team Analysis", "Update Data", "Model Performance"])
        
        if action == "Update Data":
            self._update_data_page()
        elif action == "View Predictions":
            self._predictions_page()
        elif action == "View Team Analysis":
            self._team_analysis_page()
        elif action == "Model Performance":
            self._model_performance_page()
    
    def _update_data_page(self):
        st.header("Data Update")
        
        if st.button("Scrape New Data"):
            try:
                with st.spinner("Scraping data from Premier League website..."):
                    scraper = PLScraper()
                    scraper.scrape_all_data()
                    st.success("Data successfully scraped and saved!")
            except Exception as e:
                st.error(f"Error scraping data: {str(e)}")
        
        if st.button("Retrain Model"):
            try:
                with st.spinner("Training prediction model..."):
                    processor = PLDataProcessor('pl_data.pkl')
                    predictor = PLPredictor(processor)
                    accuracy = predictor.train_model()
                    predictor.save_model()
                    st.success(f"Model trained successfully! Validation accuracy: {accuracy:.2%}")
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
    
    def _predictions_page(self):
        st.header("Match Predictions")
        
        try:
            # Load model and make predictions
            processor = PLDataProcessor('pl_data.pkl')
            predictor = PLPredictor(processor)
            
            try:
                predictor.load_model()
            except:
                st.warning("No trained model found. Training a new model...")
                predictor.train_model()
                predictor.save_model()
            
            predictions = predictor.predict_matches()
            
            if len(predictions) == 0:
                st.info("No upcoming matches found to predict.")
                return
            
            # Group matches by date
            predictions['date'] = pd.to_datetime(predictions['date'])
            predictions = predictions.sort_values('date')
            
            dates = predictions['date'].dt.date.unique()
            
            for date in dates:
                st.subheader(f"Matches on {date.strftime('%A, %d %B %Y')}")
                
                date_matches = predictions[predictions['date'].dt.date == date]
                
                for _, match in date_matches.iterrows():
                    col1, col2, col3 = st.columns([2, 1, 2])
                    
                    with col1:
                        st.markdown(f"**{match['home_team']}**")
                        st.progress(float(match['home_win_prob']))
                        st.text(f"{match['home_win_prob']:.1%}")
                    
                    with col2:
                        st.markdown("**vs**")
                        st.progress(float(match['draw_prob']))
                        st.text(f"{match['draw_prob']:.1%}")
                    
                    with col3:
                        st.markdown(f"**{match['away_team']}**")
                        st.progress(float(match['away_win_prob']))
                        st.text(f"{match['away_win_prob']:.1%}")
                    
                    # Show prediction
                    if match['home_win_prob'] > max(match['draw_prob'], match['away_win_prob']):
                        prediction = f"Prediction: **{match['home_team']} Win**"
                    elif match['away_win_prob'] > max(match['draw_prob'], match['home_win_prob']):
                        prediction = f"Prediction: **{match['away_team']} Win**"
                    else:
                        prediction = "Prediction: **Draw**"
                    
                    st.markdown(prediction)
                    st.markdown("---")
            
        except Exception as e:
            st.error(f"Error loading predictions: {str(e)}")
            st.error("Please make sure you've scraped data and trained the model first.")
    
    def _team_analysis_page(self):
        st.header("Team Analysis")
        
        try:
            processor = PLDataProcessor('pl_data.pkl')
            
            # Get teams
            teams = list(processor.teams.keys())
            selected_team = st.selectbox("Select Team", teams)
            
            if selected_team:
                st.subheader(f"Analysis for {selected_team}")
                
                # Get team results
                results = processor.results
                
                # Filter for matches involving the selected team
                team_matches = results[
                    (results['home_team'] == selected_team) | 
                    (results['away_team'] == selected_team)
                ].copy()
                
                if len(team_matches) == 0:
                    st.info(f"No match data found for {selected_team}")
                    return
                
                # Add result column from perspective of selected team
                team_matches['team_result'] = None
                
                # Home matches
                home_idx = team_matches['home_team'] == selected_team
                team_matches.loc[home_idx & (team_matches['home_score'] > team_matches['away_score']), 'team_result'] = 'Win'
                team_matches.loc[home_idx & (team_matches['home_score'] == team_matches['away_score']), 'team_result'] = 'Draw'
                team_matches.loc[home_idx & (team_matches['home_score'] < team_matches['away_score']), 'team_result'] = 'Loss'
                
                # Away matches
                away_idx = team_matches['away_team'] == selected_team
                team_matches.loc[away_idx & (team_matches['away_score'] > team_matches['home_score']), 'team_result'] = 'Win'
                team_matches.loc[away_idx & (team_matches['away_score'] == team_matches['home_score']), 'team_result'] = 'Draw'
                team_matches.loc[away_idx & (team_matches['away_score'] < team_matches['home_score']), 'team_result'] = 'Loss'
                
                # Calculate team stats
                wins = (team_matches['team_result'] == 'Win').sum()
                draws = (team_matches['team_result'] == 'Draw').sum()
                losses = (team_matches['team_result'] == 'Loss').sum()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Wins", wins)
                col2.metric("Draws", draws)
                col3.metric("Losses", losses)
                
                # Form line
                st.subheader("Recent Form")
                recent_matches = team_matches.sort_values('date', ascending=False).head(5)
                
                form_html = ""
                for _, match in recent_matches.iterrows():
                    if match['team_result'] == 'Win':
                        form_html += "🟢 "
                    elif match['team_result'] == 'Draw':
                        form_html += "🟡 "
                    else:
                        form_html += "🔴 "
                
                st.markdown(f"**Last 5 matches:** {form_html}")
                
                # Show recent match results
                st.subheader("Recent Matches")
                for _, match in recent_matches.iterrows():
                    match_date = pd.to_datetime(match['date']).strftime('%d %b %Y')
                    
                    if match['home_team'] == selected_team:
                        opponent = match['away_team']
                        score = f"{match['home_score']} - {match['away_score']}"
                        venue = "Home"
                    else:
                        opponent = match['home_team']
                        score = f"{match['away_score']} - {match['home_score']}"
                        venue = "Away"
                    
                    result_color = {
                        'Win': 'green',
                        'Draw': 'orange',
                        'Loss': 'red'
                    }
                    
                    st.markdown(
                        f"**{match_date}** vs {opponent} ({venue}): {score} - "
                        f"<span style='color:{result_color[match['team_result']]}'>{match['team_result']}</span>",
                        unsafe_allow_html=True
                    )
                
                # Show upcoming matches
                st.subheader("Upcoming Matches")
                upcoming = processor.fixtures[processor.fixtures['status'] == 'upcoming']
                team_upcoming = upcoming[
                    (upcoming['home_team'] == selected_team) | 
                    (upcoming['away_team'] == selected_team)
                ].copy()
                
                if len(team_upcoming) == 0:
                    st.info("No upcoming matches found")
                else:
                    for _, match in team_upcoming.iterrows():
                        match_date = pd.to_datetime(match['date']).strftime('%d %b %Y')
                        
                        if match['home_team'] == selected_team:
                            st.markdown(f"**{match_date}**: {selected_team} vs {match['away_team']} (Home)")
                        else:
                            st.markdown(f"**{match_date}**: {match['home_team']} vs {selected_team} (Away)")
                
                # Add visualization of goals scored/conceded
                st.subheader("Goals Analysis")
                
                # Extract goals data
                goals_data = []
                for _, match in team_matches.iterrows():
                    match_date = pd.to_datetime(match['date'])
                    
                    if match['home_team'] == selected_team:
                        goals_for = match['home_score']
                        goals_against = match['away_score']
                    else:
                        goals_for = match['away_score']
                        goals_against = match['home_score']
                    
                    goals_data.append({
                        'date': match_date,
                        'goals_for': goals_for,
                        'goals_against': goals_against
                    })
                
                goals_df = pd.DataFrame(goals_data)
                
                # Calculate rolling averages
                goals_df = goals_df.sort_values('date')
                goals_df['goals_for_avg'] = goals_df['goals_for'].rolling(5, min_periods=1).mean()
                goals_df['goals_against_avg'] = goals_df['goals_against'].rolling(5, min_periods=1).mean()
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(goals_df['date'], goals_df['goals_for_avg'], 'g-', label='Goals For (5-match avg)')
                ax.plot(goals_df['date'], goals_df['goals_against_avg'], 'r-', label='Goals Against (5-match avg)')
                ax.scatter(goals_df['date'], goals_df['goals_for'], color='green', alpha=0.5, label='Goals For')
                ax.scatter(goals_df['date'], goals_df['goals_against'], color='red', alpha=0.5, label='Goals Against')
                
                ax.set_title(f"{selected_team} - Goals Analysis")
                ax.set_xlabel('Date')
                ax.set_ylabel('Goals')
                ax.legend()
                
                # Format x-axis to show fewer date labels
                plt.xticks(rotation=45)
                
                # Improve layout
                plt.tight_layout()
                
                # Display plot
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error in team analysis: {str(e)}")
            st.error("Please make sure you've scraped data first.")
    
    def _model_performance_page(self):
        st.header("Model Performance")
        
        try:
            processor = PLDataProcessor('pl_data.pkl')
            predictor = PLPredictor(processor)
            
            try:
                predictor.load_model()
                model_loaded = True
            except:
                st.warning("No trained model found. Training a new model...")
                predictor.train_model()
                predictor.save_model()
                model_loaded = True
            
            if model_loaded and predictor.feature_importances is not None:
                # Display feature importance
                st.subheader("Feature Importance")
                
                # Plot feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    x='importance',
                    y='feature',
                    data=predictor.feature_importances,
                    ax=ax
                )
                ax.set_title('Feature Importance')
                ax.set_xlabel('Importance')
                ax.set_ylabel('Feature')
                
                st.pyplot(fig)
                
                # Cross-validation
                st.subheader("Model Validation")
                
                if st.button("Run Cross-Validation"):
                    with st.spinner("Running cross-validation..."):
                        X, y, _, _ = processor.prepare_feature_set()
                        
                        from sklearn.model_selection import cross_val_score
                        
                        # Scale features
                        X_scaled = predictor.scaler.transform(X)
                        
                        # Perform cross-validation
                        cv_scores = cross_val_score(
                            predictor.model,
                            X_scaled,
                            y,
                            cv=5,
                            scoring='accuracy'
                        )
                        
                        st.write(f"Cross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                        
                        # Plot CV scores
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.barplot(x=list(range(1, 6)), y=cv_scores, ax=ax)
                        ax.set_title('Cross-Validation Accuracy')
                        ax.set_xlabel('Fold')
                        ax.set_ylabel('Accuracy')
                        
                        st.pyplot(fig)
                
                # Confusion matrix
                st.subheader("Confusion Matrix")
                
                if st.button("Generate Confusion Matrix"):
                    with st.spinner("Generating confusion matrix..."):
                        X, y, _, _ = processor.prepare_feature_set()
                        
                        from sklearn.model_selection import train_test_split
                        from sklearn.metrics import confusion_matrix
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        
                        # Scale features
                        X_train_scaled = predictor.scaler.transform(X_train)
                        X_test_scaled = predictor.scaler.transform(X_test)
                        
                        # Train model
                        predictor.model.fit(X_train_scaled, y_train)
                        
                        # Predict
                        y_pred = predictor.model.predict(X_test_scaled)
                        
                        # Compute confusion matrix
                        cm = confusion_matrix(y_test, y_pred)
                        
                        # Plot confusion matrix
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(
                            cm, 
                            annot=True, 
                            fmt='d', 
                            cmap='Blues',
                            xticklabels=['Away Win', 'Draw', 'Home Win'],
                            yticklabels=['Away Win', 'Draw', 'Home Win'],
                            ax=ax
                        )
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('True')
                        ax.set_title('Confusion Matrix')
                        
                        st.pyplot(fig)
                        
                        # Calculate metrics
                        from sklearn.metrics import precision_score, recall_score, f1_score
                        
                        precision = precision_score(y_test, y_pred, average='weighted')
                        recall = recall_score(y_test, y_pred, average='weighted')
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Precision", f"{precision:.4f}")
                        col2.metric("Recall", f"{recall:.4f}")
                        col3.metric("F1 Score", f"{f1:.4f}")
            else:
                st.warning("Model not trained yet. Please train the model first.")
                
        except Exception as e:
            st.error(f"Error in model performance analysis: {str(e)}")
            st.error("Please make sure you've scraped data and trained the model first.")

# ---- 5. MAIN EXECUTION ----
def main():
    """Main execution function"""
    # Check if data exists, if not, scrape it
    import os
    
    if not os.path.exists('pl_data.pkl'):
        print("No data found. Scraping Premier League data...")
        scraper = PLScraper()
        scraper.scrape_all_data()
    
    # Check if model exists, if not, train it
    if not os.path.exists('pl_predictor_model.pkl'):
        print("No model found. Training prediction model...")
        processor = PLDataProcessor('pl_data.pkl')
        predictor = PLPredictor(processor)
        predictor.train_model()
        predictor.save_model()
    
    # Run the Streamlit app
    app = PLPredictorApp()
    app.run()

if __name__ == "__main__":
    main()