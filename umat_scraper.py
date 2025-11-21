from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time
from collections import deque

class JavaScriptWebScraper:
    def __init__(self, base_url, max_pages=50, headless=True):
        """
        Advanced scraper that handles JavaScript-rendered websites
        
        Args:
            base_url: Starting URL
            max_pages: Maximum pages to scrape
            headless: Run browser in headless mode (no window)
        """
        self.base_url = base_url
        self.max_pages = max_pages
        self.visited = set()
        self.scraped_data = []
        self.domain = urlparse(base_url).netloc
        
        # Setup Selenium Chrome driver
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        print("✓ Selenium Chrome driver initialized")
    
    def is_valid_url(self, url):
        """Check if URL belongs to the same domain"""
        parsed = urlparse(url)
        return parsed.netloc == self.domain and parsed.scheme in ['http', 'https']
    
    def clean_text(self, text):
        """Clean and normalize text content"""
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text
    
    def get_page_with_js(self, url):
        """Load page and wait for JavaScript to render"""
        try:
            self.driver.get(url)
            # Wait for page to load (adjust timeout as needed)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            # Extra wait for dynamic content
            time.sleep(2)
            return self.driver.page_source
        except Exception as e:
            print(f"Error loading page: {str(e)}")
            return None
    
    def extract_content(self, html, url):
        """Extract meaningful content from page"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get page title
        title = soup.find('title')
        title_text = title.get_text() if title else "No Title"
        
        # Get main content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
        
        if main_content:
            text = self.clean_text(main_content.get_text())
        else:
            text = self.clean_text(soup.get_text())
        
        # Only store pages with substantial content
        if len(text) > 100:
            return {
                'url': url,
                'title': title_text,
                'content': text[:5000],
                'length': len(text)
            }
        return None
    
    def get_links(self, html, current_url):
        """Extract all valid internal links (including JS-generated ones)"""
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        for link in soup.find_all('a', href=True):
            url = urljoin(current_url, link['href'])
            url = url.split('#')[0].split('?')[0]  # Remove fragments and queries
            
            if self.is_valid_url(url) and url not in self.visited:
                links.append(url)
        
        return links
    
    def scrape(self):
        """Main scraping function with JavaScript support"""
        queue = deque([self.base_url])
        self.visited.add(self.base_url)
        
        print(f"Starting JavaScript-aware scraping: {self.base_url}")
        print(f"Max pages: {self.max_pages}\n")
        
        try:
            while queue and len(self.scraped_data) < self.max_pages:
                url = queue.popleft()
                
                try:
                    print(f"Scraping [{len(self.scraped_data)+1}/{self.max_pages}]: {url}")
                    
                    # Load page with JavaScript rendering
                    html = self.get_page_with_js(url)
                    if not html:
                        continue
                    
                    # Extract content
                    content_data = self.extract_content(html, url)
                    if content_data:
                        self.scraped_data.append(content_data)
                        print(f"✓ Extracted {content_data['length']} characters")
                    
                    # Get new links (including JS-generated ones)
                    new_links = self.get_links(html, url)
                    for link in new_links:
                        if link not in self.visited:
                            self.visited.add(link)
                            queue.append(link)
                    
                    # Be polite
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"✗ Error scraping {url}: {str(e)}")
                    continue
        
        finally:
            # Always close the browser
            self.driver.quit()
            print("\n✓ Browser closed")
        
        print(f"✓ Scraping complete! Collected {len(self.scraped_data)} pages")
        return self.scraped_data
    
    def save_to_json(self, filename='scraped_data.json'):
        """Save scraped data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Data saved to {filename}")


# Usage Example
if __name__ == "__main__":
    scraper = JavaScriptWebScraper(
        base_url="https://www.umat.edu.gh",
        max_pages=200,  # Increase to get more pages
        headless=True  # Set to False to see the browser in action
    )
    
    # Scrape the website
    data = scraper.scrape()
    
    # Save to JSON
    scraper.save_to_json('umat_data.json')
    
    # Display summary
    print("\n" + "="*50)
    print("SCRAPING SUMMARY")
    print("="*50)
    print(f"Total pages scraped: {len(data)}")
    print(f"Total characters: {sum(d['length'] for d in data)}")
    print("\nSample titles:")
    for i, page in enumerate(data[:10], 1):
        print(f"{i}. {page['title']}")