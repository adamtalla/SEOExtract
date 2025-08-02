import logging
from typing import List
from ai_keyword_extractor import AIKeywordExtractor
from web_scraper import get_website_text_content

# Initialize the AI extractor globally
ai_extractor = AIKeywordExtractor()

def extract_keywords_from_text(text, max_keywords=10, headings_products=None):
    """
    Extract keywords from text using AI-powered analysis
    Always returns a list, never raises exceptions
    """
    keywords = []  # Initialize immediately

    try:
        if not text or not isinstance(text, str) or len(text.strip()) < 10:
            logging.warning("Text is too short or empty for keyword extraction")
            return keywords

        # Use AI extractor for intelligent keyword extraction
        keywords = ai_extractor.extract_keywords(text, max_keywords=max_keywords)

        # Apply quality filtering if headings_products provided
        if headings_products:
            keywords = filter_keywords_by_context(keywords, headings_products)

        # Ensure we don't exceed max_keywords
        return keywords[:max_keywords]

    except Exception as e:
        logging.error(f"Error in AI keyword extraction: {str(e)}")
        return extract_keywords_fallback(text, max_keywords)


def filter_keywords_by_context(keywords: List[str], context_data: List[str]) -> List[str]:
    """Filter keywords based on contextual relevance"""
    if not context_data:
        return keywords

    filtered_keywords = []
    context_lower = ' '.join(context_data).lower()

    for keyword in keywords:
        keyword_lower = keyword.lower()

        # Keep keyword if it appears in context or is semantically related
        if (keyword_lower in context_lower or
            any(word in context_lower for word in keyword_lower.split()) or
            len(keyword.split()) > 1):  # Multi-word keywords are often more specific
            filtered_keywords.append(keyword)

    # If filtering removed too many keywords, return original list
    if len(filtered_keywords) < len(keywords) * 0.3:
        return keywords

    return filtered_keywords


def extract_keywords_fallback(text, max_keywords=10, headings_products=None):
    """
    Enhanced fallback keyword extraction without external dependencies
    Always returns a list, never raises exceptions
    """
    keywords = []

    try:
        import re
        from collections import Counter

        if not text or not isinstance(text, str):
            return keywords

        # Enhanced text processing
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = re.findall(r'\b[a-z]{3,}\b', text)

        # Comprehensive stop words
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
            'were', 'been', 'being', 'are', 'is', 'it', 'as', 'if', 'do', 'did',
            'does', 'so', 'not', 'no', 'yes', 'me', 'my', 'mine', 'we', 'us',
            # Web-specific terms
            'page', 'site', 'website', 'home', 'main', 'menu', 'contact',
            'info', 'details', 'click', 'link', 'read', 'more', 'next',
            'previous', 'back', 'forward', 'start', 'end', 'section',
            'content', 'footer', 'header', 'sidebar', 'navigation', 'user',
            'account', 'login', 'logout', 'register', 'signup', 'profile',
            'search', 'results', 'result', 'submit', 'form', 'field',
            'loading', 'started', 'find', 'date', 'template', 'stuff'
        }

        # Filter words
        filtered_words = [
            word for word in words
            if (word not in stop_words and 
                len(word) >= 3 and 
                not word.isdigit() and
                not re.match(r'^(.)\1+$', word))  # Avoid repeated characters
        ]

        # Calculate word importance based on frequency and position
        word_scores = {}
        for i, word in enumerate(filtered_words):
            # Higher score for words appearing earlier
            position_weight = max(0.5, 1.0 - (i / len(filtered_words)) * 0.5)
            word_scores[word] = word_scores.get(word, 0) + position_weight

        # Extract phrases (bigrams)
        for i in range(len(filtered_words) - 1):
            phrase = f"{filtered_words[i]} {filtered_words[i+1]}"
            if len(phrase) > 6:  # Meaningful phrases
                word_scores[phrase] = word_scores.get(phrase, 0) + 2.0

        # Sort by score and return top keywords
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, score in sorted_words[:max_keywords]]

        return keywords

    except Exception as e:
        logging.error(f"Error in fallback keyword extraction: {str(e)}")
        return []


def extract_keywords_from_url(url, max_keywords=10, headings_products=None):
    """
    Extract keywords from a webpage URL using AI-powered analysis
    Always returns a list, never raises exceptions
    """
    keywords = []

    try:
        if not url or not isinstance(url, str):
            logging.warning("Invalid URL provided")
            return keywords

        # Fix URL format
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        logging.info(f"Extracting keywords from: {url}")

        # Get text content from webpage
        try:
            text_content = get_website_text_content(url)
        except Exception as e:
            logging.error(f"Error getting website text: {str(e)}")
            return get_keywords_with_requests(url, max_keywords)

        if not text_content or len(text_content.strip()) < 50:
            logging.warning(f"Insufficient content from {url}")
            return keywords

        # Use AI-powered extraction with URL context
        keywords = ai_extractor.extract_keywords(text_content, url=url, max_keywords=max_keywords)

        # Apply additional filtering if context provided
        if headings_products:
            keywords = filter_keywords_by_context(keywords, headings_products)

        logging.info(f"Extracted {len(keywords)} AI-powered keywords from {url}")
        return keywords

    except Exception as e:
        logging.error(f"Error extracting keywords from URL {url}: {str(e)}")
        return keywords


def get_keywords_with_requests(url, max_keywords=10):
    """
    Fallback method using requests and AI extraction
    Always returns a list, never raises exceptions
    """
    keywords = []

    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove scripts and styles
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text()

        # Use AI extractor even in fallback
        try:
            keywords = ai_extractor.extract_keywords(text, url=url, max_keywords=max_keywords)
        except Exception:
            keywords = extract_keywords_fallback(text, max_keywords)

        return keywords

    except Exception as e:
        logging.error(f"Error in requests fallback: {str(e)}")
        return keywords


# Test function
def test_ai_keyword_extraction():
    """Test the AI keyword extraction"""
    test_text = """
    Welcome to TechCorp, a leading artificial intelligence and machine learning company. 
    We specialize in deep learning algorithms, neural networks, and natural language processing. 
    Our data science team develops cutting-edge solutions for enterprise clients using Python, 
    TensorFlow, and cloud computing platforms. We offer consulting services, custom software 
    development, and AI model training for businesses looking to leverage artificial intelligence.
    """

    keywords = extract_keywords_from_text(test_text, 8)
    print(f"AI-extracted keywords: {keywords}")
    return keywords


if __name__ == "__main__":
    test_ai_keyword_extraction()