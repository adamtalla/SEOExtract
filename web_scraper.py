import trafilatura
import requests
from urllib.parse import urlparse
import logging
from bs4 import BeautifulSoup
import re

def get_website_text_content(url: str) -> str:
    """
    This function takes a url and returns the main text content of the website.
    The text content is extracted using trafilatura and easier to understand.
    The results is not directly readable, better to be summarized by LLM before consume
    by the user.
    """
    try:
        # Validate URL format
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format")
        
        # Send a request to the website
        downloaded = trafilatura.fetch_url(url)
        
        if not downloaded:
            raise ValueError("Failed to fetch content from the URL. The website may be inaccessible or blocking requests.")
        
        # Extract text content
        text = trafilatura.extract(downloaded)
        
        if not text:
            raise ValueError("No readable content found on the webpage")
        
        return text
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error for URL {url}: {str(e)}")
        raise ValueError(f"Failed to access the website: {str(e)}")
    except Exception as e:
        logging.error(f"Error extracting content from URL {url}: {str(e)}")
        raise ValueError(f"Error processing the webpage: {str(e)}")

def get_seo_metadata(url: str) -> dict:
    """
    Extract comprehensive SEO metadata from a webpage
    """
    try:
        # Get raw HTML
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            raise ValueError("Failed to fetch content from the URL")
        
        soup = BeautifulSoup(downloaded, 'html.parser')
        parsed_url = urlparse(url)
        
        # Extract page title
        title_tag = soup.find('title')
        page_title = title_tag.get_text().strip() if title_tag else ""
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        meta_description = meta_desc.get('content', '').strip() if meta_desc else ""
        
        # Extract headings
        headings = {}
        for i in range(1, 7):  # H1 to H6
            h_tags = soup.find_all(f'h{i}')
            headings[f'h{i}'] = [tag.get_text().strip() for tag in h_tags]
        
        # Extract URL slug
        url_slug = parsed_url.path.strip('/')
        
        # Extract image alt texts
        images = soup.find_all('img')
        image_alt_texts = [img.get('alt', '').strip() for img in images if img.get('alt')]
        
        # Count internal links
        links = soup.find_all('a', href=True)
        internal_links_count = 0
        for link in links:
            href = link['href']
            if href.startswith('/') or parsed_url.netloc in href:
                internal_links_count += 1
        
        # Check for schema markup
        schema_markup = bool(soup.find_all(attrs={'typeof': True}) or 
                           soup.find_all('script', {'type': 'application/ld+json'}))
        
        # Basic mobile-friendly check (viewport meta tag)
        viewport_meta = soup.find('meta', attrs={'name': 'viewport'})
        mobile_friendly = bool(viewport_meta)
        
        # Get main content for keyword analysis
        main_content = trafilatura.extract(downloaded) or ""
        
        return {
            'page_title': page_title,
            'meta_description': meta_description,
            'headings': headings,
            'url_slug': url_slug,
            'image_alt_texts': image_alt_texts,
            'internal_links_count': internal_links_count,
            'schema_markup': schema_markup,
            'mobile_friendly': mobile_friendly,
            'main_content': main_content,
            'page_speed': 75  # Mock score - would need real PageSpeed API
        }
        
    except Exception as e:
        logging.error(f"Error extracting SEO metadata from URL {url}: {str(e)}")
        raise ValueError(f"Error processing the webpage: {str(e)}")

def validate_url(url: str) -> bool:
    """Validate if the URL is properly formatted"""
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False
