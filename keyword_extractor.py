import yake
import logging
from web_scraper import get_website_text_content

def extract_keywords_from_text(text: str, max_keywords: int = 10) -> list:
    """
    Extract keywords from text using YAKE algorithm
    
    Args:
        text (str): The text to extract keywords from
        max_keywords (int): Maximum number of keywords to return
    
    Returns:
        list: List of extracted keywords
    """
    try:
        # Configure YAKE keyword extractor
        # Lower scores indicate better keywords
        kw_extractor = yake.KeywordExtractor(
            lan="en",                    # Language
            n=3,                         # Maximum number of words in keyphrase
            dedupLim=0.3,               # Lower deduplication threshold for better variety
            dedupFunc='seqm',           # Use sequence matcher for deduplication
            windowsSize=1,              # Window size for co-occurrence
            top=max_keywords * 2,       # Extract more to filter better ones
            features=None
        )
        
        # Extract keywords
        keywords = kw_extractor.extract_keywords(text)
        
        # Debug logging to see what YAKE returns
        logging.debug(f"YAKE extracted keywords: {keywords}")
        
        # Return only the keyword phrases (not the scores)
        # YAKE returns tuples of (keyword_phrase, score)
        result = []
        seen_keywords = set()
        
        for keyword in keywords:
            if isinstance(keyword, tuple) and len(keyword) >= 2:
                keyword_text = keyword[0].strip()
                keyword_lower = keyword_text.lower()
                
                # Filter out short keywords and duplicates
                if (len(keyword_text) >= 3 and 
                    keyword_lower not in seen_keywords and
                    not keyword_text.isdigit() and
                    len(keyword_text.split()) <= 4):  # Max 4 words
                    
                    result.append(keyword_text)
                    seen_keywords.add(keyword_lower)
                    
                    if len(result) >= max_keywords:
                        break
            else:
                keyword_text = str(keyword).strip()
                if len(keyword_text) >= 3 and keyword_text.lower() not in seen_keywords:
                    result.append(keyword_text)
                    seen_keywords.add(keyword_text.lower())
                    
                    if len(result) >= max_keywords:
                        break
        
        return result
    
    except Exception as e:
        logging.error(f"Error extracting keywords from text: {str(e)}")
        raise ValueError(f"Failed to extract keywords: {str(e)}")

def extract_keywords_from_url(url: str, max_keywords: int = 10) -> list:
    """
    Extract keywords from a webpage URL
    
    Args:
        url (str): The URL to extract keywords from
        max_keywords (int): Maximum number of keywords to return
    
    Returns:
        list: List of extracted keywords
    """
    try:
        # Get text content from the URL
        text_content = get_website_text_content(url)
        
        if not text_content or len(text_content.strip()) < 50:
            raise ValueError("Insufficient content found on the webpage for keyword extraction")
        
        # Extract keywords from the text
        keywords = extract_keywords_from_text(text_content, max_keywords)
        
        if not keywords:
            raise ValueError("No keywords could be extracted from the webpage content")
        
        return keywords
    
    except Exception as e:
        logging.error(f"Error extracting keywords from URL {url}: {str(e)}")
        raise
