import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import time

def scrape_static(url: str) -> str:
    """BeautifulSoup se static page scrape karo"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        # Unwanted tags hataao
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return text[:5000]  # First 5000 chars kaafi hai
    except Exception as e:
        return f"SCRAPE_ERROR: {str(e)}"

def scrape_dynamic(url: str) -> str:
    """Playwright se JS-heavy pages scrape karo — hardened for anti-bot"""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--disable-http2',                       # HTTP/2 protocol errors fix
                    '--disable-blink-features=AutomationControlled',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                ]
            )
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                ignore_https_errors=True,                    # Ignore cert/protocol errors
                viewport={'width': 1920, 'height': 1080},
            )
            page = context.new_page()

            # Hide webdriver / automation flags
            page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                window.chrome = { runtime: {} };
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            """)

            page.goto(url, wait_until='networkidle', timeout=30000)
            time.sleep(2)  # JS load hone do
            text = page.inner_text("body")
            browser.close()
            return text[:5000]
    except Exception as e:
        return f"DYNAMIC_SCRAPE_ERROR: {str(e)}"

def scrape_website(competitor_name: str, url: str) -> str:
    """
    Main scraper function — pehle static try karo,
    agar kam data mila toh dynamic try karo
    """
    print(f"[Scraper] Scraping {competitor_name} at {url}")

    static_data = scrape_static(url)

    # Agar static data useful lag raha hai
    if len(static_data) > 500 and "SCRAPE_ERROR" not in static_data:
        return static_data

    # Fallback: dynamic scraping
    print(f"[Scraper] Static failed, trying dynamic for {competitor_name}")
    return scrape_dynamic(url)