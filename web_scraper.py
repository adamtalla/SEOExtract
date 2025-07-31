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

def get_seo_metadata(url):
    """Extract comprehensive SEO metadata from a webpage"""
    try:
        response = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; SEOBot/1.0)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract basic metadata
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ''

        description = soup.find('meta', attrs={'name': 'description'})
        description_text = description.get('content', '').strip() if description else ''

        # Extract headings with more detail
        headings = {}
        for i in range(1, 7):
            heading_tags = soup.find_all(f'h{i}')
            if heading_tags:
                headings[f'h{i}'] = [tag.get_text().strip() for tag in heading_tags if tag.get_text().strip()]

        # Extract images with comprehensive data
        images = []
        img_tags = soup.find_all('img')
        for img in img_tags:
            src = img.get('src', '')
            if src:  # Only include images with src
                images.append({
                    'src': src,
                    'alt': img.get('alt', ''),
                    'title': img.get('title', ''),
                    'width': img.get('width', ''),
                    'height': img.get('height', '')
                })

        # Count internal and external links
        links = soup.find_all('a', href=True)
        internal_links = 0
        external_links = 0

        for link in links:
            href = link['href']
            if href.startswith('http'):
                if url in href or urlparse(url).netloc in href:
                    internal_links += 1
                else:
                    external_links += 1
            elif href.startswith('/') or not href.startswith('http'):
                internal_links += 1

        # Check for schema markup
        schema_markup = bool(soup.find('script', {'type': 'application/ld+json'})) or \
                      bool(soup.find(attrs={'itemtype': True})) or \
                      bool(soup.find(attrs={'typeof': True}))

        # Extract meta tags
        meta_tags = {}
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property') or meta.get('http-equiv')
            content = meta.get('content')
            if name and content:
                meta_tags[name.lower()] = content

        # Check for Open Graph and Twitter Card tags
        og_tags = {k: v for k, v in meta_tags.items() if k.startswith('og:')}
        twitter_tags = {k: v for k, v in meta_tags.items() if k.startswith('twitter:')}

        # Extract canonical URL
        canonical = soup.find('link', {'rel': 'canonical'})
        canonical_url = canonical.get('href') if canonical else ''

        # Check mobile viewport
        viewport = soup.find('meta', {'name': 'viewport'})
        mobile_friendly = bool(viewport and 'width=device-width' in viewport.get('content', ''))

        # Estimate page speed (basic heuristic)
        page_size = len(response.content)
        num_requests = len(soup.find_all(['script', 'link', 'img']))

        # Simple page speed heuristic (0-100)
        if page_size < 50000 and num_requests < 20:
            page_speed_score = 90
        elif page_size < 200000 and num_requests < 50:
            page_speed_score = 75
        elif page_size < 500000 and num_requests < 100:
            page_speed_score = 60
        else:
            page_speed_score = 40

        # Extract structured text content for analysis
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()

        content_text = soup.get_text()
        # Clean up whitespace
        content_text = re.sub(r'\s+', ' ', content_text).strip()

        return {
            'title': title_text,
            'description': description_text,
            'headings': headings,
            'images': images,
            'url': url,
            'internal_links_count': internal_links,
            'external_links_count': external_links,
            'schema_markup': schema_markup,
            'meta_tags': meta_tags,
            'og_tags': og_tags,
            'twitter_tags': twitter_tags,
            'canonical_url': canonical_url,
            'mobile_friendly': mobile_friendly,
            'page_speed_score': page_speed_score,
            'content_text': content_text,
            'page_size': page_size,
            'total_requests': num_requests
        }

    except Exception as e:
        print(f"Error extracting SEO metadata: {str(e)}")
        return {
            'title': '',
            'description': '',
            'headings': {},
            'images': [],
            'url': url,
            'internal_links_count': 0,
            'external_links_count': 0,
            'schema_markup': False,
            'meta_tags': {},
            'og_tags': {},
            'twitter_tags': {},
            'canonical_url': '',
            'mobile_friendly': False,
            'page_speed_score': 0,
            'content_text': '',
            'page_size': 0,
            'total_requests': 0
        }

def validate_url(url: str) -> bool:
    """Validate if the URL is properly formatted"""
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False