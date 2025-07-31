
import re
from typing import Dict, List, Tuple, Any
from urllib.parse import urlparse

class SEOAuditor:
    def __init__(self):
        self.priority_weights = {
            'critical': 10,
            'high': 7,
            'medium': 4,
            'low': 2
        }
    
    def analyze_page(self, seo_data: Dict, keywords: List[str], content: str) -> List[Dict]:
        """
        Perform comprehensive SEO audit based on page data
        """
        audit_results = []
        
        # Title analysis
        audit_results.extend(self._analyze_title(seo_data.get('title', ''), keywords))
        
        # Meta description analysis
        audit_results.extend(self._analyze_meta_description(seo_data.get('description', ''), keywords))
        
        # Heading structure analysis
        audit_results.extend(self._analyze_headings(seo_data.get('headings', {}), keywords))
        
        # URL analysis
        audit_results.extend(self._analyze_url(seo_data.get('url', ''), keywords))
        
        # Image analysis
        audit_results.extend(self._analyze_images(seo_data.get('images', [])))
        
        # Content analysis
        audit_results.extend(self._analyze_content(content, keywords))
        
        # Keyword analysis
        audit_results.extend(self._analyze_keywords(keywords, content))
        
        # Technical SEO
        audit_results.extend(self._analyze_technical_seo(seo_data))
        
        # Sort by priority
        audit_results.sort(key=lambda x: self.priority_weights.get(x['priority'].lower(), 0), reverse=True)
        
        return audit_results
    
    def _analyze_title(self, title: str, keywords: List[str]) -> List[Dict]:
        """Analyze page title for SEO issues"""
        issues = []
        
        if not title:
            issues.append({
                'type': 'Title',
                'priority': 'Critical',
                'issue': 'Missing page title',
                'description': 'The page has no title tag, which is crucial for SEO and user experience.',
                'impact': 'Search engines cannot understand the page topic, severely impacting rankings.',
                'recommendation': 'Add a descriptive title tag (50-60 characters) that includes your primary keyword.',
                'order': 1
            })
            return issues
        
        title_length = len(title)
        
        if title_length > 60:
            issues.append({
                'type': 'Title',
                'priority': 'High',
                'issue': f'Title too long ({title_length} characters)',
                'description': 'The page title exceeds the recommended 60 characters and may be truncated in search results.',
                'impact': 'Truncated titles reduce click-through rates and keyword visibility.',
                'recommendation': f'Shorten the title to under 60 characters. Current: "{title[:50]}..."',
                'order': 2
            })
        elif title_length < 30:
            issues.append({
                'type': 'Title',
                'priority': 'Medium',
                'issue': f'Title too short ({title_length} characters)',
                'description': 'The page title is shorter than recommended, missing opportunities for keyword inclusion.',
                'impact': 'Short titles may not provide enough context for search engines and users.',
                'recommendation': 'Expand the title to 50-60 characters while including relevant keywords.',
                'order': 4
            })
        
        # Check if primary keywords are in title
        if keywords:
            primary_keyword = keywords[0].lower()
            if primary_keyword not in title.lower():
                issues.append({
                    'type': 'Title',
                    'priority': 'High',
                    'issue': 'Primary keyword not in title',
                    'description': f'The primary keyword "{keywords[0]}" is not present in the page title.',
                    'impact': 'Missing primary keywords in titles significantly reduces ranking potential.',
                    'recommendation': f'Include "{keywords[0]}" naturally in the title, preferably near the beginning.',
                    'order': 3
                })
        
        return issues
    
    def _analyze_meta_description(self, description: str, keywords: List[str]) -> List[Dict]:
        """Analyze meta description for SEO issues"""
        issues = []
        
        if not description:
            issues.append({
                'type': 'Meta Description',
                'priority': 'High',
                'issue': 'Missing meta description',
                'description': 'The page has no meta description tag.',
                'impact': 'Search engines will generate their own snippet, which may not be compelling.',
                'recommendation': 'Add a compelling meta description (150-160 characters) that includes primary keywords.',
                'order': 5
            })
            return issues
        
        desc_length = len(description)
        
        if desc_length > 160:
            issues.append({
                'type': 'Meta Description',
                'priority': 'Medium',
                'issue': f'Meta description too long ({desc_length} characters)',
                'description': 'The meta description exceeds 160 characters and may be truncated.',
                'impact': 'Truncated descriptions reduce click-through rates and message clarity.',
                'recommendation': f'Shorten to 150-160 characters. Current: "{description[:80]}..."',
                'order': 6
            })
        elif desc_length < 120:
            issues.append({
                'type': 'Meta Description',
                'priority': 'Low',
                'issue': f'Meta description too short ({desc_length} characters)',
                'description': 'The meta description could be longer to provide more context.',
                'impact': 'Missing opportunities to include compelling copy and keywords.',
                'recommendation': 'Expand to 150-160 characters with more descriptive content.',
                'order': 8
            })
        
        # Check for keyword inclusion
        if keywords:
            primary_keyword = keywords[0].lower()
            if primary_keyword not in description.lower():
                issues.append({
                    'type': 'Meta Description',
                    'priority': 'Medium',
                    'issue': 'Primary keyword not in meta description',
                    'description': f'The primary keyword "{keywords[0]}" is not in the meta description.',
                    'impact': 'Missing keywords reduce relevance signals to search engines.',
                    'recommendation': f'Include "{keywords[0]}" naturally in the meta description.',
                    'order': 7
                })
        
        return issues
    
    def _analyze_headings(self, headings: Dict, keywords: List[str]) -> List[Dict]:
        """Analyze heading structure for SEO issues"""
        issues = []
        
        h1_tags = headings.get('h1', [])
        h1_count = len(h1_tags)
        
        if h1_count == 0:
            issues.append({
                'type': 'Headings',
                'priority': 'High',
                'issue': 'Missing H1 tag',
                'description': 'The page has no H1 tag.',
                'impact': 'H1 tags help search engines understand the main topic of the page.',
                'recommendation': 'Add one H1 tag that includes your primary keyword and describes the main topic.',
                'order': 9
            })
        elif h1_count > 1:
            issues.append({
                'type': 'Headings',
                'priority': 'High',
                'issue': f'Multiple H1 tags ({h1_count} found)',
                'description': f'The page has {h1_count} H1 tags: {", ".join(h1_tags[:3])}{"..." if len(h1_tags) > 3 else ""}',
                'impact': 'Multiple H1s can confuse search engines about the main page topic.',
                'recommendation': 'Use only one H1 tag for the main topic. Convert others to H2 or H3 tags.',
                'order': 10
            })
        
        # Check if H1 contains keywords
        if h1_tags and keywords:
            h1_text = ' '.join(h1_tags).lower()
            primary_keyword = keywords[0].lower()
            if primary_keyword not in h1_text:
                issues.append({
                    'type': 'Headings',
                    'priority': 'Medium',
                    'issue': 'Primary keyword not in H1',
                    'description': f'The H1 tag doesn\'t contain the primary keyword "{keywords[0]}".',
                    'impact': 'H1 tags with keywords help establish page relevance.',
                    'recommendation': f'Include "{keywords[0]}" naturally in the H1 tag.',
                    'order': 11
                })
        
        # Check heading hierarchy
        all_headings = []
        for level in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            if level in headings:
                all_headings.extend([(level, text) for text in headings[level]])
        
        if len(all_headings) < 3:
            issues.append({
                'type': 'Headings',
                'priority': 'Low',
                'issue': 'Limited heading structure',
                'description': f'The page has only {len(all_headings)} heading tags.',
                'impact': 'Poor heading structure makes content harder to scan and understand.',
                'recommendation': 'Add more H2 and H3 tags to structure your content better.',
                'order': 12
            })
        
        return issues
    
    def _analyze_url(self, url: str, keywords: List[str]) -> List[Dict]:
        """Analyze URL structure for SEO issues"""
        issues = []
        
        if not url:
            return issues
        
        parsed_url = urlparse(url)
        path = parsed_url.path
        
        # Check URL length
        if len(url) > 100:
            issues.append({
                'type': 'URL',
                'priority': 'Low',
                'issue': f'Long URL ({len(url)} characters)',
                'description': 'The URL is longer than recommended.',
                'impact': 'Long URLs are harder to share and may be truncated.',
                'recommendation': 'Consider shortening the URL structure and removing unnecessary parameters.',
                'order': 13
            })
        
        # Check for keywords in URL
        if keywords and path:
            primary_keyword = keywords[0].lower().replace(' ', '-')
            if primary_keyword not in path.lower():
                issues.append({
                    'type': 'URL',
                    'priority': 'Medium',
                    'issue': 'Primary keyword not in URL',
                    'description': f'The URL doesn\'t contain the primary keyword "{keywords[0]}".',
                    'impact': 'Keywords in URLs provide relevance signals to search engines.',
                    'recommendation': f'Consider including "{primary_keyword}" in the URL path.',
                    'order': 14
                })
        
        # Check for URL best practices
        if '_' in path:
            issues.append({
                'type': 'URL',
                'priority': 'Low',
                'issue': 'Underscores in URL',
                'description': 'The URL contains underscores instead of hyphens.',
                'impact': 'Search engines prefer hyphens over underscores for word separation.',
                'recommendation': 'Replace underscores with hyphens in URLs.',
                'order': 15
            })
        
        return issues
    
    def _analyze_images(self, images: List[Dict]) -> List[Dict]:
        """Analyze images for SEO issues"""
        issues = []
        
        if not images:
            return issues
        
        missing_alt = sum(1 for img in images if not img.get('alt', '').strip())
        
        if missing_alt > 0:
            issues.append({
                'type': 'Images',
                'priority': 'Medium',
                'issue': f'{missing_alt} images missing alt text',
                'description': f'{missing_alt} out of {len(images)} images lack descriptive alt text.',
                'impact': 'Missing alt text hurts accessibility and reduces keyword relevance.',
                'recommendation': 'Add descriptive alt text to all images, naturally incorporating relevant keywords.',
                'order': 16
            })
        
        # Check for keyword usage in alt text
        alt_texts = ' '.join([img.get('alt', '') for img in images]).lower()
        if alt_texts and len(alt_texts) > 20:  # Only check if there's substantial alt text
            keyword_usage = sum(1 for keyword in keywords[:3] if keyword.lower() in alt_texts)
            if keyword_usage == 0:
                issues.append({
                    'type': 'Images',
                    'priority': 'Low',
                    'issue': 'Keywords not used in alt text',
                    'description': 'Image alt text doesn\'t include any target keywords.',
                    'impact': 'Missing opportunity to reinforce page topic relevance.',
                    'recommendation': 'Include relevant keywords naturally in image alt text where appropriate.',
                    'order': 17
                })
        
        return issues
    
    def _analyze_content(self, content: str, keywords: List[str]) -> List[Dict]:
        """Analyze content for SEO issues"""
        issues = []
        
        if not content:
            issues.append({
                'type': 'Content',
                'priority': 'Critical',
                'issue': 'No content found',
                'description': 'The page appears to have no readable content.',
                'impact': 'Search engines need substantial content to understand and rank pages.',
                'recommendation': 'Add substantial, valuable content (at least 300 words) relevant to your keywords.',
                'order': 18
            })
            return issues
        
        word_count = len(content.split())
        
        if word_count < 300:
            issues.append({
                'type': 'Content',
                'priority': 'Medium',
                'issue': f'Thin content ({word_count} words)',
                'description': 'The page has less than 300 words of content.',
                'impact': 'Thin content may be seen as low-quality by search engines.',
                'recommendation': 'Expand content to at least 300-500 words with valuable, relevant information.',
                'order': 19
            })
        
        return issues
    
    def _analyze_keywords(self, keywords: List[str], content: str) -> List[Dict]:
        """Analyze keyword usage and density"""
        issues = []
        
        if not keywords or not content:
            return issues
        
        content_lower = content.lower()
        content_words = len(content.split())
        
        for i, keyword in enumerate(keywords[:5]):  # Check top 5 keywords
            keyword_lower = keyword.lower()
            
            # Skip nonsensical keywords
            if self._is_nonsensical_keyword(keyword):
                issues.append({
                    'type': 'Keywords',
                    'priority': 'Low',
                    'issue': f'Nonsensical keyword: "{keyword}"',
                    'description': f'The keyword "{keyword}" appears to be irrelevant or nonsensical.',
                    'impact': 'Targeting irrelevant keywords wastes SEO efforts.',
                    'recommendation': f'Remove "{keyword}" from keyword targeting and focus on relevant terms.',
                    'order': 20 + i
                })
                continue
            
            # Check keyword presence
            if keyword_lower not in content_lower:
                priority = 'High' if i == 0 else 'Medium' if i < 3 else 'Low'
                issues.append({
                    'type': 'Keywords',
                    'priority': priority,
                    'issue': f'Keyword "{keyword}" not found in content',
                    'description': f'The target keyword "{keyword}" doesn\'t appear in the page content.',
                    'impact': 'Keywords not present in content reduce topical relevance.',
                    'recommendation': f'Naturally incorporate "{keyword}" into the page content.',
                    'order': 25 + i
                })
            else:
                # Calculate keyword density
                keyword_count = content_lower.count(keyword_lower)
                density = (keyword_count / content_words) * 100
                
                if density > 3:  # Over-optimization
                    issues.append({
                        'type': 'Keywords',
                        'priority': 'Medium',
                        'issue': f'Keyword "{keyword}" over-optimized ({density:.1f}% density)',
                        'description': f'The keyword "{keyword}" appears {keyword_count} times ({density:.1f}% density).',
                        'impact': 'Keyword stuffing can result in search engine penalties.',
                        'recommendation': f'Reduce usage of "{keyword}" to 1-2% density for natural flow.',
                        'order': 30 + i
                    })
        
        return issues
    
    def _analyze_technical_seo(self, seo_data: Dict) -> List[Dict]:
        """Analyze technical SEO aspects"""
        issues = []
        
        # Check for schema markup
        if not seo_data.get('schema_markup', False):
            issues.append({
                'type': 'Technical SEO',
                'priority': 'Low',
                'issue': 'No schema markup detected',
                'description': 'The page doesn\'t appear to have structured data markup.',
                'impact': 'Schema markup helps search engines understand content better.',
                'recommendation': 'Add relevant schema markup (e.g., Article, Organization, Product) to enhance search visibility.',
                'order': 35
            })
        
        # Check page speed
        page_speed = seo_data.get('page_speed_score', 0)
        if page_speed > 0:
            if page_speed < 50:
                issues.append({
                    'type': 'Technical SEO',
                    'priority': 'High',
                    'issue': f'Poor page speed ({page_speed}/100)',
                    'description': 'The page loads slowly, affecting user experience.',
                    'impact': 'Slow pages have higher bounce rates and lower search rankings.',
                    'recommendation': 'Optimize images, enable compression, and minimize CSS/JavaScript to improve speed.',
                    'order': 36
                })
            elif page_speed < 75:
                issues.append({
                    'type': 'Technical SEO',
                    'priority': 'Medium',
                    'issue': f'Average page speed ({page_speed}/100)',
                    'description': 'The page speed could be improved.',
                    'impact': 'Better page speed improves user experience and search rankings.',
                    'recommendation': 'Optimize images and enable browser caching to improve load times.',
                    'order': 37
                })
        
        # Check mobile friendliness
        if not seo_data.get('mobile_friendly', True):
            issues.append({
                'type': 'Technical SEO',
                'priority': 'High',
                'issue': 'Not mobile-friendly',
                'description': 'The page is not optimized for mobile devices.',
                'impact': 'Mobile-unfriendly pages rank poorly in mobile search results.',
                'recommendation': 'Implement responsive design to ensure mobile compatibility.',
                'order': 38
            })
        
        return issues
    
    def _is_nonsensical_keyword(self, keyword: str) -> bool:
        """Check if a keyword appears to be nonsensical or irrelevant"""
        nonsensical_patterns = [
            r'loading\s+date',
            r'templates\s+loading',
            r'\d{4}-\d{2}-\d{2}',  # Dates
            r'click\s+here',
            r'read\s+more',
            r'learn\s+more\s+about',
            r'www\.',
            r'\.com',
            r'http[s]?://',
        ]
        
        keyword_lower = keyword.lower()
        
        # Check against patterns
        for pattern in nonsensical_patterns:
            if re.search(pattern, keyword_lower):
                return True
        
        # Check for overly long phrases (likely extraction errors)
        if len(keyword.split()) > 5:
            return True
        
        # Check for repeated words
        words = keyword_lower.split()
        if len(words) != len(set(words)) and len(words) > 2:
            return True
        
        return False

    def format_audit_report(self, audit_results: List[Dict]) -> str:
        """Format audit results into a readable report"""
        if not audit_results:
            return "No SEO issues found. Your page appears to be well-optimized!"
        
        report = "## SEO Audit Report\n\n"
        report += f"Found {len(audit_results)} SEO issues and opportunities:\n\n"
        
        priority_groups = {}
        for result in audit_results:
            priority = result['priority']
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(result)
        
        # Order by priority
        priority_order = ['Critical', 'High', 'Medium', 'Low']
        
        for priority in priority_order:
            if priority in priority_groups:
                report += f"### {priority} Priority Issues\n\n"
                for i, result in enumerate(priority_groups[priority], 1):
                    report += f"**{i}. {result['type']}: {result['issue']}**\n\n"
                    report += f"*Problem:* {result['description']}\n\n"
                    report += f"*Impact:* {result['impact']}\n\n"
                    report += f"*Recommendation:* {result['recommendation']}\n\n"
                    report += "---\n\n"
        
        return report
