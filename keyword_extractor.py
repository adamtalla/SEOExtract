import yake
import logging
from web_scraper import get_website_text_content


def extract_keywords_from_text(text, max_keywords=10):
    """
    Extract keywords from text using YAKE algorithm
    Always returns a list, never raises exceptions
    """
    keywords = []  # Initialize immediately

    try:
        # Check if we have yake
        import yake

        if not text or not isinstance(text, str) or len(text.strip()) < 10:
            logging.warning(
                "Text is too short or empty for keyword extraction")
            return keywords

        # Configure YAKE
        kw_extractor = yake.KeywordExtractor(lan="en",
                                             n=3,
                                             dedupLim=0.3,
                                             dedupFunc='seqm',
                                             windowsSize=1,
                                             top=max_keywords * 2,
                                             features=None)

        # Extract keywords
        yake_keywords = kw_extractor.extract_keywords(text)

        # Process results
        seen_keywords = set()
        for keyword in yake_keywords:
            try:
                if isinstance(keyword, tuple) and len(keyword) >= 2:
                    keyword_text = str(keyword[0]).strip()
                else:
                    keyword_text = str(keyword).strip()

                keyword_lower = keyword_text.lower()

                if (len(keyword_text) >= 3
                        and keyword_lower not in seen_keywords
                        and not keyword_text.isdigit()
                        and len(keyword_text.split()) <= 4):

                    keywords.append(keyword_text)
                    seen_keywords.add(keyword_lower)

                    if len(keywords) >= max_keywords:
                        break
            except Exception as e:
                logging.error(f"Error processing individual keyword: {str(e)}")
                continue

        return keywords

    except ImportError:
        logging.warning(
            "YAKE not available, using fallback keyword extraction")
        return extract_keywords_fallback(text, max_keywords)
    except Exception as e:
        logging.error(f"Error in YAKE keyword extraction: {str(e)}")
        return extract_keywords_fallback(text, max_keywords)


def extract_keywords_fallback(text, max_keywords=10):
    """
    Fallback keyword extraction without YAKE
    Always returns a list, never raises exceptions
    """
    keywords = []  # Initialize immediately

    try:
        import re
        from collections import Counter

        if not text or not isinstance(text, str):
            return keywords

        # Simple text processing
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = re.findall(r'\b[a-z]{3,}\b', text)

        # Basic stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'within',
            'without', 'this', 'that', 'these', 'those', 'what', 'which',
            'who', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own',
            'same', 'than', 'very', 'can', 'will', 'just', 'should', 'now',
            'get', 'has', 'had', 'have', 'him', 'his', 'her', 'she', 'its',
            'our', 'out', 'you', 'your', 'they', 'them', 'their', 'was',
            'were', 'been', 'being', 'are'
        }

        # Filter and count
        filtered_words = [
            word for word in words if word not in stop_words and len(word) > 2
        ]
        word_counts = Counter(filtered_words)

        # Get top words
        keywords = [
            word for word, count in word_counts.most_common(max_keywords)
        ]

        return keywords

    except Exception as e:
        logging.error(f"Error in fallback keyword extraction: {str(e)}")
        return keywords


def extract_keywords_from_url(url, max_keywords=10):
    """
    Extract keywords from a webpage URL
    Always returns a list, never raises exceptions
    """
    keywords = []  # Initialize immediately

    try:
        if not url or not isinstance(url, str):
            logging.warning("Invalid URL provided")
            return keywords

        # Fix URL format
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        logging.info(f"Extracting keywords from: {url}")

        # Try to get text content
        try:
            from web_scraper import get_website_text_content
            text_content = get_website_text_content(url)
        except ImportError:
            logging.error("web_scraper module not available")
            return get_keywords_with_requests(url, max_keywords)
        except Exception as e:
            logging.error(f"Error getting website text: {str(e)}")
            return get_keywords_with_requests(url, max_keywords)

        if not text_content or len(text_content.strip()) < 50:
            logging.warning(f"Insufficient content from {url}")
            return keywords

        # Extract keywords from text
        keywords = extract_keywords_from_text(text_content, max_keywords)

        logging.info(f"Extracted {len(keywords)} keywords from {url}")
        return keywords

    except Exception as e:
        logging.error(f"Error extracting keywords from URL {url}: {str(e)}")
        return keywords


def get_keywords_with_requests(url, max_keywords=10):
    """
    Fallback method using requests directly
    Always returns a list, never raises exceptions
    """
    keywords = []  # Initialize immediately

    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text()
        keywords = extract_keywords_fallback(text, max_keywords)

        return keywords

    except Exception as e:
        logging.error(f"Error in requests fallback: {str(e)}")
        return keywords


# Simple test function
def test_keyword_extraction():
    """Test the keyword extraction"""
    test_text = "This is a test about web development and Python programming. SEO optimization is important for websites."
    keywords = extract_keywords_from_text(test_text, 5)
    print(f"Test keywords: {keywords}")
    return keywords


if __name__ == "__main__":
    test_keyword_extraction()
