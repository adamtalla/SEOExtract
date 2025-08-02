import yake
import trafilatura
import requests
import logging
from typing import List, Dict, Optional
from urllib.parse import urlparse, urljoin
import time
import re

# Import training system
try:
    from seo_keyword_trainer import get_keyword_trainer
    TRAINER_AVAILABLE = True
except ImportError:
    TRAINER_AVAILABLE = False
    logging.warning("SEO keyword trainer not available")

import logging
from typing import List
from ai_keyword_extractor import AIKeywordExtractor
from web_scraper import get_website_text_content

# Initialize the AI extractor globally
ai_extractor = AIKeywordExtractor()

def extract_keywords_from_text(text, max_keywords=10, headings_products=None):
    """
    Extract keywords from text using enhanced AI-powered analysis
    Always returns a list, never raises exceptions
    """
    keywords = []  # Initialize immediately

    try:
        if not text or not isinstance(text, str) or len(text.strip()) < 10:
            logging.warning("Text is too short or empty for keyword extraction")
            return keywords

        # Use enhanced AI extractor for intelligent keyword extraction
        keywords = ai_extractor.extract_keywords(text, max_keywords=max_keywords)

        # Apply enhanced filtering for SEO relevance
        if keywords:
            keywords = filter_seo_keywords(keywords, text)

        # Apply context filtering if headings_products provided
        if headings_products and keywords:
            keywords = filter_keywords_by_context(keywords, headings_products)

        # Final quality filter - remove any remaining generic terms
        keywords = [kw for kw in keywords if not is_generic_term(kw)]

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
def filter_seo_keywords(keywords: List[str], text: str) -> List[str]:
    """Filter keywords for SEO relevance and quality"""
    if not keywords:
        return []

    filtered_keywords = []
    text_lower = text.lower()

    for keyword in keywords:
        keyword_lower = keyword.lower().strip()

        # Skip if too short or contains numbers only
        if len(keyword_lower) < 3 or keyword_lower.isdigit():
            continue

        # Check frequency - keyword should appear but not too often
        frequency = text_lower.count(keyword_lower)
        text_length = len(text.split())
        frequency_ratio = frequency / text_length * 100 if text_length > 0 else 0

        # Skip if appears too rarely or too frequently (keyword stuffing)
        if frequency_ratio < 0.1 or frequency_ratio > 5.0:
            continue

        # Prefer multi-word phrases as they're more specific
        word_count = len(keyword.split())
        if word_count >= 2 or (word_count == 1 and len(keyword) >= 5):
            filtered_keywords.append(keyword)

    return filtered_keywords


def is_generic_term(keyword: str) -> bool:
    """Check if keyword is too generic for SEO value"""
    keyword_lower = keyword.lower().strip()

    # Generic business terms that don't add SEO value
    generic_terms = {
        'welcome', 'about', 'company', 'business', 'team', 'staff', 'people',
        'work', 'working', 'offer', 'provide', 'giving', 'making', 'help',
        'helping', 'looking', 'find', 'finding', 'get', 'getting', 'take',
        'taking', 'use', 'using', 'try', 'trying', 'want', 'wanting',
        'need', 'needing', 'call', 'calling', 'contact', 'today', 'now',
        'time', 'years', 'experience', 'quality', 'best', 'great', 'good',
        'excellent', 'amazing', 'perfect', 'professional', 'experienced'
    }

    # Single generic words
    if keyword_lower in generic_terms:
        return True

    # Generic phrases
    generic_phrases = [
        'call now', 'contact us', 'get started', 'learn more', 'find out',
        'years experience', 'quality service', 'best service', 'great service',
        'professional service', 'call today', 'contact today'
    ]

    if keyword_lower in generic_phrases:
        return True

    return False


def get_detailed_keywords(text: str, url: str = "", max_keywords: int = 10):
    """Get detailed keyword analysis with classification and scoring"""
    try:
        from keyword_result import EnhancedKeywordExtractor
        enhanced_extractor = EnhancedKeywordExtractor()
        return enhanced_extractor.extract_keywords_detailed(text, url, max_keywords)
    except Exception as e:
        logging.error(f"Error in detailed keyword extraction: {str(e)}")
        # Fallback to simple extraction
        simple_keywords = extract_keywords_from_text(text, max_keywords)
        return [{'keyword': kw, 'score': 1.0, 'category': 'general'} for kw in simple_keywords]


def test_ai_keyword_extraction():
    """Test the enhanced AI keyword extraction"""
    test_text = """
    Welcome to NYC Plumbing Solutions, your trusted emergency plumber in Manhattan. 
    We provide 24-hour plumbing services including drain cleaning, pipe repair, 
    water heater installation, and bathroom renovation. Our licensed plumbers 
    offer affordable residential and commercial plumbing services throughout 
    New York City. Call us for emergency plumbing repair, toilet installation, 
    sink repair, and professional plumbing maintenance. We serve Manhattan, 
    Brooklyn, Queens, and the Bronx with same-day plumbing services.
    """

    # Test simple extraction
    keywords = extract_keywords_from_text(test_text, 8)
    print(f"Enhanced AI-extracted keywords: {keywords}")

    # Test detailed extraction
    try:
        detailed_results = get_detailed_keywords(test_text, "https://nycplumbing.com", 8)
        print(f"Detailed results: {[r.keyword for r in detailed_results]}")
        print(f"Categories: {set(r.category for r in detailed_results)}")
    except Exception as e:
        print(f"Detailed extraction error: {e}")

    return keywords


if __name__ == "__main__":
    test_ai_keyword_extraction()