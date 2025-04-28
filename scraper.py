import os
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium_stealth import stealth
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException

# ========== Setup Paths ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'get_data', 'data')
os.makedirs(DATA_DIR, exist_ok=True)  # create data dir if it doesn't exist

# ========== Setup Selenium Driver (just once) ==========
options = webdriver.ChromeOptions()
# options.add_argument('--headless')  # Uncomment this if you want headless mode
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument('--start-maximized')
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115 Safari/537.36')

# Create driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Set stealth settings
stealth(driver,
    languages=["en-US", "en"],
    vendor="Google Inc.",
    platform="Win32",
    webgl_vendor="Intel Inc.",
    renderer="Intel Iris OpenGL Engine",
    fix_hairline=True,
)

# Set page timeout
driver.set_page_load_timeout(30)

# ========== Scraping Setup ==========
seasons_wished = list(reversed(range(2020, 2024)))

root_url = 'https://www.oddsportal.com/soccer/england/premier-league'
rest_of_url = '/results/'
main_url = root_url + rest_of_url
seasons_url = [f"{root_url}-{season}-{season + 1}{rest_of_url}" for season in seasons_wished]
all_urls = [main_url] + seasons_url

df_list = []  # faster than using df.append inside loop

def is_empty(col):
    try:
        return col.text.strip()
    except:
        return None

def get_active_page_number(soup):
    try:
        return soup.find('span', class_='active-page').text
    except AttributeError:
        return None

# ========== Scraping Loop ==========
for url in all_urls:
    print(f"Processing URL: {url}")
    try:
        driver.get(url)
        time.sleep(5)  # Increased wait time to ensure page loads fully
        
        # Handle cookies popup if it appears
        try:
            accept_cookies = driver.find_element(By.ID, "onetrust-accept-btn-handler")
            driver.execute_script("arguments[0].click();", accept_cookies)
            time.sleep(1)
        except:
            pass  # No cookies banner or already accepted
        
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        
        page_num = 1
        while True:
            print(f"Processing page {page_num} of {url}")
            previous_page = get_active_page_number(soup)
            
            # Extract the current season
            try:
                season = soup.find_all('span', class_='active')[1].text
                print(f"Current season: {season}")
            except (IndexError, AttributeError) as e:
                print(f"Error finding season: {e}")
                season = "Unknown"
            
            # Process each match row
            for col in soup.find_all('tr', attrs={'deactivate': True}):
                try:
                    try:
                        match_date = col.findPrevious('th', attrs={'class': 'center nob-border'}).text.strip()
                        if match_date.endswith("Show odds"):
                            match_date = match_date[:-9]  # Remove "Show odds" text
                    except:
                        match_date = "Unknown"
                        
                    match_name = col.find('td', class_='name table-participant').text.replace('\xa0', '') if col.find('td', class_='name table-participant') else "Unknown"
                    match_result = col.find('td', class_='center bold table-odds table-score').text if col.find('td', class_='center bold table-odds table-score') else "Unknown"

                    odds = col.find_all('td', class_='odds-nowrp')
                    h_odd = is_empty(odds[0]) if len(odds) > 0 else None
                    d_odd = is_empty(odds[1]) if len(odds) > 1 else None
                    a_odd = is_empty(odds[2]) if len(odds) > 2 else None

                    df_list.append({
                        'season': season,
                        'date': match_date,
                        'match_name': match_name,
                        'result': match_result,
                        'h_odd': h_odd,
                        'd_odd': d_odd,
                        'a_odd': a_odd,
                    })
                    
                except Exception as e:
                    print(f"Error parsing row: {e}")

            print(f'Page {page_num} done!')
            page_num += 1

            # Check if there's a next page
            try:
                # First try to find the next page button
                next_button = driver.find_element(By.XPATH, "//a[contains(text(), '»')]")
                driver.execute_script("arguments[0].scrollIntoView();", next_button)
                time.sleep(1)  # Wait a bit after scrolling
                driver.execute_script("arguments[0].click();", next_button)
                time.sleep(3)  # Increased wait time after clicking
                
                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                new_page = get_active_page_number(soup)
                
                # Check if we actually navigated to a new page
                if not new_page or (previous_page and new_page == previous_page):
                    print("No new page loaded, ending pagination")
                    break
                    
            except (NoSuchElementException, ElementClickInterceptedException) as e:
                print(f"No next button found or couldn't click it: {e}")
                break
                
    except TimeoutException:
        print(f"Timeout loading {url}, moving to next URL")
        continue
    except Exception as e:
        print(f"Error processing {url}: {e}")
        continue

    print(f"URL {url} done!")

# Clean up
driver.quit()
print('Scraping finished!')

# ========== Save Data ==========
if df_list:
    df = pd.DataFrame(df_list)
    df = df[['season', 'date', 'match_name', 'result', 'h_odd', 'd_odd', 'a_odd']]  # reordering columns
    df.to_csv(os.path.join(DATA_DIR, 'matches.csv'), index=False)
    print(f'Data saved to {os.path.join(DATA_DIR, "matches.csv")}')
    print(f'Total matches scraped: {len(df)}')
else:
    print("No data was collected. Check for possible website structure changes or blocking.")