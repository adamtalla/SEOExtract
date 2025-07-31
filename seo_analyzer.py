
import re
from collections import Counter

def analyze_keyword_usage(content: str, target_keywords: list) -> dict:
    """Analyze how keywords are used in the content"""
    content_lower = content.lower()
    keyword_usage = {}
    
    for keyword in target_keywords:
        keyword_lower = keyword.lower()
        # Count exact matches and partial matches
        exact_count = len(re.findall(r'\b' + re.escape(keyword_lower) + r'\b', content_lower))
        keyword_usage[keyword] = exact_count
    
    return keyword_usage

def generate_seo_suggestions(seo_data: dict, target_keywords: list) -> list:
    """
    Generate customized SEO suggestions based on actual website data
    """
    suggestions = []
    
    # Analyze page title
    page_title = seo_data.get('page_title', '')
    if not page_title:
        suggestions.append("The page is missing a title tag. Add a descriptive title (50-60 characters) that includes your primary keyword to improve search engine visibility.")
    elif len(page_title) < 30:
        suggestions.append(f"The page title is too short ({len(page_title)} characters). Expand it to 50-60 characters and include your primary keyword for better SEO impact.")
    elif len(page_title) > 60:
        suggestions.append(f"The page title is too long ({len(page_title)} characters). Shorten it to under 60 characters to prevent truncation in search results.")
    elif target_keywords and not any(keyword.lower() in page_title.lower() for keyword in target_keywords):
        suggestions.append("The page title doesn't include your primary keywords. Add your main keyword naturally to the title to improve relevance for search engines.")
    
    # Analyze meta description
    meta_description = seo_data.get('meta_description', '')
    if not meta_description:
        suggestions.append("The page is missing a meta description. Add a compelling description (150-160 characters) that includes your primary keyword to improve click-through rates from search results.")
    elif len(meta_description) < 120:
        suggestions.append(f"The meta description is too short ({len(meta_description)} characters). Expand it to 150-160 characters to provide more context and include relevant keywords.")
    elif len(meta_description) > 160:
        suggestions.append(f"The meta description is too long ({len(meta_description)} characters). Shorten it to under 160 characters to prevent truncation in search results.")
    
    # Analyze headings
    headings = seo_data.get('headings', {})
    h1_tags = headings.get('h1', [])
    if not h1_tags:
        suggestions.append("The page is missing an H1 tag. Add a clear, keyword-rich H1 heading that describes the main topic of the page.")
    elif len(h1_tags) > 1:
        suggestions.append(f"The page has {len(h1_tags)} H1 tags. Use only one H1 per page and structure content with H2, H3 tags for better SEO hierarchy.")
    elif target_keywords and h1_tags and not any(keyword.lower() in h1_tags[0].lower() for keyword in target_keywords):
        suggestions.append("The H1 heading doesn't include your primary keywords. Incorporate your main keyword naturally into the H1 for better topical relevance.")
    
    # Check heading structure
    h2_tags = headings.get('h2', [])
    if len(h2_tags) == 0 and len(seo_data.get('main_content', '')) > 500:
        suggestions.append("The page lacks H2 subheadings. Break up long content with descriptive H2 tags to improve readability and SEO structure.")
    
    # Analyze URL structure
    url_slug = seo_data.get('url_slug', '')
    if url_slug and target_keywords:
        slug_has_keywords = any(keyword.lower().replace(' ', '-') in url_slug.lower() for keyword in target_keywords)
        if not slug_has_keywords:
            suggestions.append("The URL doesn't include your target keywords. Consider using a more descriptive URL slug that includes your primary keyword.")
    
    # Analyze images
    image_alt_texts = seo_data.get('image_alt_texts', [])
    if not image_alt_texts:
        suggestions.append("Images on the page are missing alt text. Add descriptive alt text to all images, including relevant keywords where appropriate, to improve accessibility and SEO.")
    elif target_keywords:
        alt_has_keywords = any(any(keyword.lower() in alt.lower() for keyword in target_keywords) for alt in image_alt_texts)
        if not alt_has_keywords:
            suggestions.append("Image alt texts don't include your target keywords. Add relevant keywords to image alt text naturally to improve topical relevance.")
    
    # Analyze internal linking
    internal_links_count = seo_data.get('internal_links_count', 0)
    if internal_links_count == 0:
        suggestions.append("The page has no internal links. Add links to related pages on your site to improve navigation and distribute page authority.")
    elif internal_links_count < 3:
        suggestions.append(f"The page has only {internal_links_count} internal links. Add more links to relevant pages to improve site navigation and SEO value.")
    
    # Analyze schema markup
    if not seo_data.get('schema_markup', False):
        suggestions.append("The page lacks structured data (schema markup). Add relevant schema markup (e.g., Article, Product, Organization) to enhance search result appearance with rich snippets.")
    
    # Analyze mobile friendliness
    if not seo_data.get('mobile_friendly', False):
        suggestions.append("The page is missing a viewport meta tag. Add <meta name='viewport' content='width=device-width, initial-scale=1'> to ensure mobile-friendly display.")
    
    # Analyze page speed (mock analysis)
    page_speed = seo_data.get('page_speed', 0)
    if page_speed < 50:
        suggestions.append(f"The page speed score is low ({page_speed}/100). Optimize images, leverage browser caching, and minimize CSS/JavaScript to improve loading times.")
    elif page_speed < 80:
        suggestions.append(f"The page speed score could be improved ({page_speed}/100). Consider optimizing images and enabling compression to reach 80+ for better user experience and SEO.")
    
    # Keyword density analysis
    main_content = seo_data.get('main_content', '')
    if target_keywords and main_content:
        keyword_usage = analyze_keyword_usage(main_content, target_keywords)
        word_count = len(main_content.split())
        
        for keyword, count in keyword_usage.items():
            if count == 0:
                suggestions.append(f"The target keyword '{keyword}' doesn't appear in the page content. Include it naturally throughout the text to improve relevance.")
            elif word_count > 0:
                density = (count / word_count) * 100
                if density > 3:
                    suggestions.append(f"The keyword '{keyword}' appears too frequently ({density:.1f}% density). Reduce usage to avoid keyword stuffing penalties.")
                elif density < 0.5 and word_count > 200:
                    suggestions.append(f"The keyword '{keyword}' has low density ({density:.1f}%). Use it more naturally throughout the content to improve topical relevance.")
    
    # Content length analysis
    content_length = len(main_content.split()) if main_content else 0
    if content_length < 300:
        suggestions.append(f"The page content is quite short ({content_length} words). Consider expanding to 300+ words to provide more value and improve search rankings.")
    
    return suggestions[:10]  # Return top 10 most relevant suggestions
