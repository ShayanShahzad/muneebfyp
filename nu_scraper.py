import os
import json
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError

# Keep track of visited URLs
visited = set()
results = []

def clean_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style']):
        tag.decompose()
    return soup.get_text(separator='\n', strip=True)

def is_internal_link(link, domain):
    try:
        parsed = urlparse(link)
        return (parsed.netloc == '' or domain in parsed.netloc) and not parsed.fragment
    except:
        return False

def scrape_page(page, url):
    try:
        print(f"🔄 Visiting: {url}")
        page.goto(url, timeout=30000)
        page.wait_for_timeout(2000)  # Give JS time to load

        html = page.content()
        text = clean_text(html)

        print(f"✅ Scraped: {url}")
        return {"url": url, "content": text}
    except TimeoutError:
        print(f"❌ Timeout on {url}")
    except Exception as e:
        print(f"❌ Error scraping {url}: {str(e)}")
    return None

def crawl_site(start_url):
    print("🚀 Starting browser...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        domain = urlparse(start_url).netloc
        to_visit = [start_url]

        print(f"🌐 Domain to crawl: {domain}\n")

        while to_visit:
            current_url = to_visit.pop(0)
            if current_url in visited:
                continue
            visited.add(current_url)

            data = scrape_page(page, current_url)
            if data:
                results.append(data)

            try:
                anchors = page.locator("a")
                links = anchors.evaluate_all("elements => elements.map(el => el.href)")
                print(f"🔍 Found {len(links)} links on: {current_url}")

                for link in links:
                    if is_internal_link(link, domain) and link not in visited and link not in to_visit:
                        to_visit.append(link)

            except Exception as e:
                print(f"⚠️ Failed to extract links on {current_url}: {str(e)}")

        browser.close()
    print("\n✅ Crawling complete.")

# ========= Run Script =========
if __name__ == "__main__":
    start_url = "https://www.nu.edu.pk/"
    crawl_site(start_url)

    output_file = "nu_all_pages_playwright.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n📁 Saved {len(results)} pages to {output_file}")
