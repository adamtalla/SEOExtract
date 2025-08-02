import yake
import logging
from web_scraper import get_website_text_content


def extract_keywords_from_text(text, max_keywords=10, headings_products=None):
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


        # Advanced filtering logic
        seen_keywords = set()
        low_quality_terms = {
            'loading', 'started', 'click', 'find', 'date', 'template', 'stuff', 'thing', 'content',
            'page', 'site', 'website', 'main', 'menu', 'contact', 'info', 'details', 'read', 'next',
            'previous', 'back', 'forward', 'start', 'end', 'top', 'bottom', 'section', 'footer', 'header',
            'sidebar', 'navigation', 'user', 'account', 'login', 'logout', 'register', 'signup', 'sign',
            'profile', 'search', 'results', 'result', 'submit', 'form', 'field', 'input', 'output', 'data',
            'value', 'values', 'number', 'numbers', 'list', 'item', 'items', 'example', 'examples', 'sample',
            'samples', 'case', 'cases', 'type', 'types', 'day', 'days', 'week', 'weeks', 'month', 'months',
            'year', 'years', 'buy', 'now', 'best', 'price', 'make', 'money', 'fast', 'work', 'home', 'earn',
            'free', 'trial', 'special', 'offer', 'instant', 'access', 'credit', 'card', 'limited', 'time',
            'act', 'miracle', 'secret', 'method', 'product', 'items', 'something', 'object', 'elements',
            'everything', 'anything', 'feature', 'solution', 'service', 'app', 'tool', 'site', 'place',
            'password', 'admin', 'root', 'token', 'key', 'api', 'secret', 'test', 'demo', 'fake', 'unknown',
            'placeholder', 'lorem', 'ipsum', 'dummy', 'sample', 'unknown', 'blah', 'content', 'abc', 'xyz',
            'randomtext', 'gibberish', '123abc', 'abc123', '0000', '1111', '9999', 'xxx', 'sex', 'hot',
            'girls', 'porn', 'naked', 'adult', '18+', 'webcam', 'nude', 'escort', 'chat', 'erotic', 'nsfw',
            'dating', 'singles', '!!!', '???', '$$$', '###', '@@@', '%%%', '^^^', '&&&', '***', '///', '---',
            '===', '+++', '[[[', ']]]', '(((', ')))', '{{', '}}', ':::', ';;;', '...',
        }
        vague_single_words = {'project', 'template', 'tools'}
        valid_short = {'ai', 'ux', 'seo', 'api', 'css', 'js', 'ux', 'ui'}
        block_substrings = ["loading", "started", "date", "find", "click", "submit"]
        import re
        def is_gibberish(word):
            if len(set(word)) <= 2:
                return True
            if re.fullmatch(r'[a-z]{4,}', word) and sum(1 for c in word if c in 'aeiou') < 1:
                return True
            if re.fullmatch(r'(.)\1{2,}', word):
                return True
            return False

        def is_incomplete_verb_phrase(phrase):
            # e.g. "router to create", "start to", "try to", "or"/"to" mid-phrase
            words = phrase.lower().split()
            if len(words) >= 2:
                if words[0] in {'start', 'try', 'use', 'find', 'click', 'submit', 'make', 'get', 'set', 'go', 'run', 'build', 'create', 'add', 'remove', 'update', 'delete', 'edit', 'open', 'close', 'move', 'change', 'select', 'choose', 'show', 'hide', 'enable', 'disable', 'install', 'uninstall', 'download', 'upload', 'save', 'load', 'import', 'export', 'search', 'filter', 'sort', 'view', 'see', 'test', 'check', 'verify', 'review', 'plan', 'organize', 'manage', 'setup', 'configure', 'connect', 'disconnect', 'register', 'login', 'logout', 'sign', 'submit', 'apply', 'join', 'leave', 'share', 'invite', 'send', 'receive', 'read', 'write', 'print', 'scan', 'copy', 'paste', 'cut', 'replace', 'merge', 'split', 'sync', 'backup', 'restore', 'reset', 'restart', 'shutdown', 'power', 'charge', 'play', 'pause', 'stop', 'record', 'watch', 'listen', 'speak', 'talk', 'call', 'message', 'email', 'post', 'tweet', 'like', 'follow', 'unfollow', 'subscribe', 'unsubscribe', 'block', 'report', 'flag', 'rate', 'review', 'comment', 'vote', 'support', 'help', 'assist', 'guide', 'teach', 'learn', 'study', 'train', 'practice', 'test', 'debug', 'fix', 'patch', 'update', 'upgrade', 'downgrade', 'install', 'uninstall', 'download', 'upload', 'sync', 'backup', 'restore', 'reset', 'restart', 'shutdown', 'power', 'charge', 'play', 'pause', 'stop', 'record', 'watch', 'listen', 'speak', 'talk', 'call', 'message', 'email', 'post', 'tweet', 'like', 'follow', 'unfollow', 'subscribe', 'unsubscribe', 'block', 'report', 'flag', 'rate', 'review', 'comment', 'vote', 'support', 'help', 'assist', 'guide', 'teach', 'learn', 'study', 'train', 'practice', 'test', 'debug', 'fix', 'patch', 'update', 'upgrade', 'downgrade'}:
                    # If phrase is just verb + preposition or verb + "to"/"or" + ...
                    if len(words) < 3:
                        return True
                    if words[1] in {'to', 'or'}:
                        return True
            # Block if "or" or "to" in the middle and phrase is incomplete
            if any(w in {'or', 'to'} for w in words[1:-1]):
                return True
            return False

        def passes_whitelist(phrase):
            if not headings_products:
                return True  # If no whitelist provided, allow all
            phrase_lower = phrase.lower().strip()
            for hp in headings_products:
                if phrase_lower in hp.lower():
                    return True
            return False

        for keyword in yake_keywords:
            try:
                if isinstance(keyword, tuple) and len(keyword) >= 2:
                    keyword_text = str(keyword[0]).strip()
                else:
                    keyword_text = str(keyword).strip()

                keyword_lower = keyword_text.lower()
                # Remove numbers, special chars, and filter by rules
                if (
                    keyword_lower not in seen_keywords
                    and keyword_lower not in low_quality_terms
                    and not keyword_text.isdigit()
                    and not re.fullmatch(r'\W+', keyword_text)
                    and not re.fullmatch(r'\d+', keyword_text)
                    and not is_gibberish(keyword_lower)
                    and (
                        (len(keyword_text) > 2)
                        or (keyword_lower in valid_short)
                    )
                    and len(keyword_text.split()) <= 4
                    and not any(sub in keyword_lower for sub in block_substrings)
                    and not (len(keyword_text.split()) == 1 and keyword_lower in vague_single_words)
                    and not is_incomplete_verb_phrase(keyword_text)
                    and passes_whitelist(keyword_text)
                ):
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


def extract_keywords_fallback(text, max_keywords=10, headings_products=None):
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
            # Standard English stop words
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
            'does', 'so', 'not', 'no', 'yes', 'i', 'me', 'my', 'mine', 'we', 'us',
            'because', 'once', 'over', 'under', 'again', 'off', 'then', 'there',
            'here', 'also', 'too', 'much', 'many', 'every', 'each', 'per', 'via',
            'may', 'might', 'must', 'shall', 'would', 'could', 'like', 'even',
            'made', 'make', 'let', 'see', 'seen', 'used', 'using', 'use', 'want',
            'needs', 'needed', 'since', 'due', 'yet', 'still', 'however', 'thus',
            'therefore', 'etc', 'etc.', 'etcetera', 'said', 'says', 'saying',
            # Web and generic terms
            'page', 'site', 'website', 'webpage', 'home', 'main', 'menu', 'contact',
            'info', 'information', 'details', 'click', 'link', 'read', 'more', 'next',
            'previous', 'back', 'forward', 'start', 'started', 'end', 'top', 'bottom',
            'section', 'content', 'footer', 'header', 'sidebar', 'navigation', 'user',
            'account', 'login', 'logout', 'register', 'signup', 'sign', 'profile',
            'search', 'results', 'result', 'submit', 'form', 'field', 'input', 'output',
            'data', 'value', 'values', 'number', 'numbers', 'list', 'item', 'items',
            'example', 'examples', 'sample', 'samples', 'case', 'cases', 'type', 'types',
            'day', 'days', 'week', 'weeks', 'month', 'months', 'year', 'years',
            # Add more as needed for your domain
            # User-provided stop words and phrases
            'asdf', 'qwerty', 'test', 'example', 'demo', 'fake', 'lorem', 'ipsum', 'dummy', 'sample', 'unknown', 'blah', 'thing', 'stuff', 'content', 'placeholder',
            'lol', 'omg', 'wtf', 'lmao', 'noob', 'hey', 'hi', 'pls', 'okay', 'ok', 'yeah', 'nah', 'bruh', 'yo', 'dude', 'idk', 'smh', 'yolo', 'chill', 'whatever', 'cool', 'sup',
            'buy', 'now', 'best', 'price', 'click', 'here', 'make', 'money', 'fast', 'work', 'home', 'earn', 'free', 'trial', 'special', 'offer', 'instant', 'access', 'credit', 'card', 'limited', 'time', 'act', 'miracle', 'secret', 'method',
            'product', 'items', 'something', 'object', 'elements', 'everything', 'anything', 'feature', 'solution', 'service', 'app', 'tool', 'site', 'place',
            'abc', 'xyz', 'lkj', 'asdfgh', 'qwertyuiop', 'zxcvbnm', 'bbbb', 'kkkk', 'xxxx', 'yyyy', 'tttt', 'randomtext', 'gibberish', '123abc', 'abc123', '0000', '1111', '9999',
            'xxx', 'sex', 'hot', 'girls', 'porn', 'naked', 'adult', '18+', 'webcam', 'nude', 'escort', 'chat', 'erotic', 'nsfw', 'dating', 'singles',
            '!!!', '???', '$$$', '###', '@@@', '%%%', '^^^', '&&&', '***', '///', '---', '===', '+++', '[[[', ']]]', '(((', ')))', '{{', '}}', ':::', ';;;', '...',
            'password', 'admin', 'login', 'user', 'root', '123456', 'letmein', 'welcome', 'iloveyou', 'token', 'key', 'api', 'secret',
            # Common short stop words (repeated for completeness)
            'the', 'is', 'to', 'of', 'and', 'a', 'in', 'on', 'with', 'for', 'from', 'by', 'it', 'be', 'at', 'or', 'as', 'if', 'an', 'not', 'so', 'too',
        }

        # Advanced filtering logic for fallback
        low_quality_terms = {
            'loading', 'started', 'click', 'find', 'date', 'template', 'stuff', 'thing', 'content',
            'page', 'site', 'website', 'main', 'menu', 'contact', 'info', 'details', 'read', 'next',
            'previous', 'back', 'forward', 'start', 'end', 'top', 'bottom', 'section', 'footer', 'header',
            'sidebar', 'navigation', 'user', 'account', 'login', 'logout', 'register', 'signup', 'sign',
            'profile', 'search', 'results', 'result', 'submit', 'form', 'field', 'input', 'output', 'data',
            'value', 'values', 'number', 'numbers', 'list', 'item', 'items', 'example', 'examples', 'sample',
            'samples', 'case', 'cases', 'type', 'types', 'day', 'days', 'week', 'weeks', 'month', 'months',
            'year', 'years', 'buy', 'now', 'best', 'price', 'make', 'money', 'fast', 'work', 'home', 'earn',
            'free', 'trial', 'special', 'offer', 'instant', 'access', 'credit', 'card', 'limited', 'time',
            'act', 'miracle', 'secret', 'method', 'product', 'items', 'something', 'object', 'elements',
            'everything', 'anything', 'feature', 'solution', 'service', 'app', 'tool', 'site', 'place',
            'password', 'admin', 'root', 'token', 'key', 'api', 'secret', 'test', 'demo', 'fake', 'unknown',
            'placeholder', 'lorem', 'ipsum', 'dummy', 'sample', 'unknown', 'blah', 'content', 'abc', 'xyz',
            'randomtext', 'gibberish', '123abc', 'abc123', '0000', '1111', '9999', 'xxx', 'sex', 'hot',
            'girls', 'porn', 'naked', 'adult', '18+', 'webcam', 'nude', 'escort', 'chat', 'erotic', 'nsfw',
            'dating', 'singles', '!!!', '???', '$$$', '###', '@@@', '%%%', '^^^', '&&&', '***', '///', '---',
            '===', '+++', '[[[', ']]]', '(((', ')))', '{{', '}}', ':::', ';;;', '...',
        }

        valid_short = {'ai', 'ux', 'seo', 'api', 'css', 'js', 'ux', 'ui'}
        vague_single_words = {'project', 'template', 'tools'}
        block_substrings = ["loading", "started", "date", "find", "click", "submit"]
        def is_gibberish(word):
            if len(set(word)) <= 2:
                return True
            if re.fullmatch(r'[a-z]{4,}', word) and sum(1 for c in word if c in 'aeiou') < 1:
                return True
            if re.fullmatch(r'(.)\1{2,}', word):
                return True
            return False

        def is_incomplete_verb_phrase(phrase):
            words = phrase.lower().split()
            if len(words) >= 2:
                if words[0] in {'start', 'try', 'use', 'find', 'click', 'submit', 'make', 'get', 'set', 'go', 'run', 'build', 'create', 'add', 'remove', 'update', 'delete', 'edit', 'open', 'close', 'move', 'change', 'select', 'choose', 'show', 'hide', 'enable', 'disable', 'install', 'uninstall', 'download', 'upload', 'save', 'load', 'import', 'export', 'search', 'filter', 'sort', 'view', 'see', 'test', 'check', 'verify', 'review', 'plan', 'organize', 'manage', 'setup', 'configure', 'connect', 'disconnect', 'register', 'login', 'logout', 'sign', 'submit', 'apply', 'join', 'leave', 'share', 'invite', 'send', 'receive', 'read', 'write', 'print', 'scan', 'copy', 'paste', 'cut', 'replace', 'merge', 'split', 'sync', 'backup', 'restore', 'reset', 'restart', 'shutdown', 'power', 'charge', 'play', 'pause', 'stop', 'record', 'watch', 'listen', 'speak', 'talk', 'call', 'message', 'email', 'post', 'tweet', 'like', 'follow', 'unfollow', 'subscribe', 'unsubscribe', 'block', 'report', 'flag', 'rate', 'review', 'comment', 'vote', 'support', 'help', 'assist', 'guide', 'teach', 'learn', 'study', 'train', 'practice', 'test', 'debug', 'fix', 'patch', 'update', 'upgrade', 'downgrade', 'install', 'uninstall', 'download', 'upload', 'sync', 'backup', 'restore', 'reset', 'restart', 'shutdown', 'power', 'charge', 'play', 'pause', 'stop', 'record', 'watch', 'listen', 'speak', 'talk', 'call', 'message', 'email', 'post', 'tweet', 'like', 'follow', 'unfollow', 'subscribe', 'unsubscribe', 'block', 'report', 'flag', 'rate', 'review', 'comment', 'vote', 'support', 'help', 'assist', 'guide', 'teach', 'learn', 'study', 'train', 'practice', 'test', 'debug', 'fix', 'patch', 'update', 'upgrade', 'downgrade'}:
                    if len(words) < 3:
                        return True
                    if words[1] in {'to', 'or'}:
                        return True
            if any(w in {'or', 'to'} for w in words[1:-1]):
                return True
            return False

        def passes_whitelist(phrase):
            if not headings_products:
                return True
            phrase_lower = phrase.lower().strip()
            for hp in headings_products:
                if phrase_lower in hp.lower():
                    return True
            return False

        filtered_words = [
            word for word in words
            if word not in stop_words
            and word not in low_quality_terms
            and (len(word) > 2 or word in valid_short)
            and not word.isdigit()
            and not is_gibberish(word)
            and not re.fullmatch(r'\W+', word)
            and not re.fullmatch(r'\d+', word)
            and not any(sub in word for sub in block_substrings)
            and not (len(word.split()) == 1 and word in vague_single_words)
            and not is_incomplete_verb_phrase(word)
            and passes_whitelist(word)
        ]
        word_counts = Counter(filtered_words)

        keywords = [
            word for word, count in word_counts.most_common(max_keywords)
        ]

        return keywords

    except Exception as e:
        logging.error(f"Error in fallback keyword extraction: {str(e)}")
        return keywords


def extract_keywords_from_url(url, max_keywords=10, headings_products=None):
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


        # Try to extract headings/products for whitelist logic
        # Placeholder: extract from web_scraper if available, else pass through param
        extracted_headings_products = None
        try:
            from web_scraper import get_website_headings_and_products
            extracted_headings_products = get_website_headings_and_products(url)
        except Exception:
            extracted_headings_products = headings_products

        # Extract keywords from text with whitelist
        keywords = extract_keywords_from_text(text_content, max_keywords, headings_products=extracted_headings_products)

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
