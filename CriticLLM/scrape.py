import time
import random
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def perform_random_delay(delay=3, random_offset=0.5):
    """Pause execution for a short randomized delay."""
    actual_delay = delay + random.uniform(0, random_offset)
    logger.debug(f"Waiting for {actual_delay:.2f} seconds")
    time.sleep(actual_delay)

def extract_ign_review(soup):
    """Extract review text from an IGN page."""
    logger.debug("Attempting to extract IGN review")
    container = soup.find("div", {"data-cy": "article-content"})
    if container is None:
        logger.debug("Primary container not found, trying fallback")
        container = soup.find("div", {"id": "article-body"})
    if container:
        elements = container.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        texts = [elem.get_text(separator=" ", strip=True) for elem in elements if elem.get_text(strip=True)]
        text_length = sum(len(t) for t in texts)
        logger.info(f"Successfully extracted IGN review with {text_length} characters")
        return "\n".join(texts)
    logger.warning("No review content found in IGN page")
    return None

def extract_pcgamer_review(soup):
    """Extract review text from a PCGamer page."""
    logger.debug("Attempting to extract PCGamer review")
    container = soup.find("div", {"id": "article-body"})
    if not container:
        logger.debug("Primary container not found, trying fallback")
        container = soup.find("div", class_="article-body")
    if container:
        elements = container.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        texts = [elem.get_text(separator=" ", strip=True) for elem in elements if elem.get_text(strip=True)]
        text_length = sum(len(t) for t in texts)
        logger.info(f"Successfully extracted PCGamer review with {text_length} characters")
        return "\n".join(texts)
    logger.warning("No review content found in PCGamer page")
    return None

def extract_eurogamer_review(soup):
    """Extract review text from a Eurogamer page."""
    logger.debug("Attempting to extract Eurogamer review")
    container = soup.find("div", {"data-component": "article-content", "class": "article_body"})
    if container is None:
        logger.debug("Primary container not found, trying fallback")
        container = soup.find("section")
    if container:
        elements = container.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        texts = [elem.get_text(separator=" ", strip=True) for elem in elements if elem.get_text(strip=True)]
        text_length = sum(len(t) for t in texts)
        logger.info(f"Successfully extracted Eurogamer review with {text_length} characters")
        return "\n".join(texts)
    logger.warning("No review content found in Eurogamer page")
    return None

def extract_vg247_review(soup):
    """Extract review text from a VG247 page."""
    logger.debug("Attempting to extract VG247 review")
    container = soup.find("div", {"data-component": "article-content", "class": "article_body"})
    if container is None:
        logger.debug("Primary container not found, trying fallback")
        container = soup.find("div", class_="article_body")
    if container:
        elements = container.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        texts = [elem.get_text(separator=" ", strip=True) for elem in elements if elem.get_text(strip=True)]
        text_length = sum(len(t) for t in texts)
        logger.info(f"Successfully extracted VG247 review with {text_length} characters")
        return "\n".join(texts)
    logger.warning("No review content found in VG247 page")
    return None

def scrape_critic_reviews(game_name, headers=None):
    """
    Scrapes critic reviews for a given game from four websites.
    Returns a DataFrame with columns: 'website', 'review_text', 'url'.
    """
    logger.info(f"Starting review scraping for game: {game_name}")

    if headers is None:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9"
        }
    logger.debug(f"Using headers: {headers}")

    websites = {
        "IGN": {
            "url": "https://www.ign.com/articles/{slug}-review",
            "extractor": extract_ign_review
        },
        "PCGamer": {
            "url": "https://www.pcgamer.com/{slug}-review/",
            "extractor": extract_pcgamer_review
        },
        "Eurogamer": {
            "url": "https://www.eurogamer.net/articles/{slug}-review/",
            "extractor": extract_eurogamer_review
        },
        "VG247": {
            "url": "https://www.vg247.com/{slug}-review/",
            "extractor": extract_vg247_review
        }
    }

    reviews_list = []
    slug = game_name.lower().replace(" ", "-").replace("'", "")
    logger.info(f"Generated URL slug: {slug}")

    for site_name, site_info in websites.items():
        url = site_info['url'].format(slug=slug)
        logger.info(f"Attempting to scrape {site_name} from URL: {url}")

        try:
            logger.debug(f"Sending GET request to {url}")
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                logger.warning(f"Failed to retrieve review from {site_name}. HTTP Status: {response.status_code}")
                continue

            logger.debug(f"Successfully retrieved page from {site_name}")
            soup = BeautifulSoup(response.text, 'html.parser')
            review_text = site_info["extractor"](soup)

            if not review_text:
                logger.warning(f"No review text extracted from {site_name}")
                continue

            reviews_list.append({
                "website": site_name,
                "review_text": review_text,
                "url": url
            })
            logger.info(f"Successfully scraped review from {site_name}")

            perform_random_delay(1, 0.3)

        except requests.Timeout:
            logger.error(f"Timeout while scraping {site_name}")
        except requests.RequestException as e:
            logger.error(f"Network error while scraping {site_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error while scraping {site_name}: {str(e)}", exc_info=True)

    df = pd.DataFrame(reviews_list)
    logger.info(f"Scraping complete. Retrieved {len(df)} reviews")
    return df
