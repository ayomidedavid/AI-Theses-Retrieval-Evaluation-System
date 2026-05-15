import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote

BASE_URL = "http://repository.pgcollege.ui.edu.ng:8080/xmlui/"
DOWNLOAD_DIR = "downloaded_pdfs"

def download_file(url, folder):
    """Downloads a file from a given URL to a specified folder"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Try to extract the filename from the URL
        # e.g. /bitstream/handle/123456789/1/Document.pdf?sequence=1
        parsed_url_path = unquote(url.split('?')[0])
        filename = parsed_url_path.split('/')[-1]
        
        # Verify it's actually a PDF file
        if not filename.lower().endswith('.pdf'):
            # In cases where URL doesn't have .pdf, we can check the Content-Type header
            if 'application/pdf' not in response.headers.get('Content-Type', ''):
                return # Skip non-PDF files
            filename += '.pdf'
                
        filepath = os.path.join(folder, filename)
        
        if os.path.exists(filepath):
            print(f"[-] Already exists, skipping: {filename}")
            return
            
        print(f"[+] Downloading: {filename}...")
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
    except Exception as e:
        print(f"[!] Failed to download {url}. Error: {e}")

def get_page_links(url):
    """Fetches all links from a single HTML page"""
    try:
        html = requests.get(url, timeout=30).text
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        for a in soup.find_all('a', href=True):
            links.append(urljoin(BASE_URL, a['href']))
        return list(set(links))
    except Exception as e:
        print(f"[!] Error fetching page {url}: {e}")
        return []

def scrape_repository(start_url, max_pages=15000):
    """Crawls the DSpace repository using Breadth-First Search"""
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        
    visited_urls = set()
    urls_to_visit = [start_url]
    
    print(f"Starting Scraper on: {start_url}")
    print(f"PDFs will be saved to: {os.path.abspath(DOWNLOAD_DIR)}\n")
    
    pages_crawled = 0
    
    while urls_to_visit and pages_crawled < max_pages:
        current_url = urls_to_visit.pop(0)
        
        if current_url in visited_urls:
            continue
            
        visited_urls.add(current_url)
        
        # Skip login/admin pages and language variants
        if any(skip in current_url for skip in ['/login', '/register', '/password-login', 'locale-attribute']):
            continue

        # Is it a download link for a file (bitstream)?
        if '/bitstream/' in current_url:
            if '.pdf' in current_url.lower():
                download_file(current_url, DOWNLOAD_DIR)
                time.sleep(1) # Small delay to avoid overloading the server
            continue
            
        # Is it a standard DSpace item/collection/community page?
        # We only want to extract links from structural pages.
        if "/handle/" in current_url or current_url == BASE_URL:
            print(f"[*] Crawling page: {current_url}")
            pages_crawled += 1
            
            links = get_page_links(current_url)
            
            for link in links:
                # We only want internal XMLUI links that we haven't visited
                if BASE_URL in link and link not in visited_urls:
                    # Ignore sorting & browse query parameters which create infinite duplicate loops
                    if "?" in link and "/bitstream/" not in link:
                        if "offset=" not in link:
                            continue 
                    
                    # Prioritize downloading PDFs immediately (Depth-First)
                    if "/bitstream/" in link and ".pdf" in link.lower():
                        urls_to_visit.insert(0, link)
                    else:
                        urls_to_visit.append(link)
                    
            time.sleep(0.5) # Crawler delay so we don't accidentally attack the server

if __name__ == "__main__":
    # You can change START_URL to a specific collection (e.g., Agriculture) to limit the scan
    # Example: START_URL = "http://repository.pgcollege.ui.edu.ng:8080/xmlui/handle/123456789/3"
    START_URL = BASE_URL
    scrape_repository(START_URL)
    print("\nScraping complete!")
