import trafilatura
import requests
from urllib.parse import urlparse
import logging

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

def validate_url(url: str) -> bool:
    """Validate if the URL is properly formatted"""
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False
