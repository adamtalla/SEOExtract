import os
import json
import logging
from typing import Dict, List
import openai

class AISEOAnalyzer:
    def __init__(self):
        """Initialize the AI SEO Analyzer with OpenAI API key"""
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if self.openai_api_key:
            openai.api_key = self.openai_api_key

    def generate_ai_suggestions(self, url: str, seo_data: Dict, keywords: List[str]) -> List[Dict]:
        """
        Generate AI-powered SEO suggestions based on actual website data
        """
        if not self.openai_api_key:
            logging.warning("OpenAI API key not found, falling back to rule-based suggestions")
            return self._fallback_suggestions(seo_data, keywords)

        try:
            # Prepare the prompt with actual data
            prompt = self._build_analysis_prompt(url, seo_data, keywords)

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert SEO analyst who provides specific, actionable recommendations based on actual website data. Always return valid JSON."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=1500,
                temperature=0.3
            )

            # Parse the response
            suggestions_text = response.choices[0].message.content.strip()

            # Clean up the response if it has markdown formatting
            if suggestions_text.startswith('```json'):
                suggestions_text = suggestions_text.replace('```json', '').replace('```', '').strip()

            # Parse JSON response
            suggestions = json.loads(suggestions_text)

            # Validate and clean suggestions
            return self._validate_suggestions(suggestions)

        except Exception as e:
            logging.error(f"Error generating AI suggestions: {str(e)}")
            return self._fallback_suggestions(seo_data, keywords)

    def generate_hybrid_suggestions(self, url: str, seo_data: Dict, keywords: List[str], plan: str) -> List[Dict]:
        """
        Generate hybrid suggestions using AI for premium users, fallback for others
        """
        # Ensure keywords is always a list
        if not isinstance(keywords, list):
            keywords = []

        if plan == 'premium' and self.openai_api_key:
            try:
                ai_suggestions = self.generate_ai_suggestions(url, seo_data, keywords)
                if ai_suggestions:
                    return ai_suggestions
            except Exception as e:
                logging.error(f"AI suggestions failed: {str(e)}")

        # Fallback to rule-based suggestions
        return self._fallback_suggestions(seo_data, keywords)

    def _build_analysis_prompt(self, url: str, seo_data: Dict, keywords: List[str]) -> str:
        """Build a comprehensive analysis prompt for OpenAI"""

        # Ensure keywords is a list
        if not isinstance(keywords, list):
            keywords = []

        # Extract key data for analysis
        title = seo_data.get('title', 'No title found')
        description = seo_data.get('description', 'No meta description found')
        headings = seo_data.get('headings', {})
        content_text = seo_data.get('content_text', '')

        # Format headings
        headings_text = ""
        for level, heading_list in headings.items():
            if heading_list:
                headings_text += f"{level.upper()}: {', '.join(heading_list[:3])}; "

        # Get content sample (first 500 chars)
        content_sample = content_text[:500] + "..." if len(content_text) > 500 else content_text

        prompt = f"""You are a professional SEO expert. Given detailed information about a webpage, you analyze its SEO factors and provide clear, actionable, and prioritized suggestions to improve its search engine ranking. 

Input data includes:

- Page title
- Meta description
- Headings (H1, H2, H3, etc.)
- Main content text
- List of target keywords for the page
- Technical SEO info such as page speed score and mobile friendliness

Your task:

1. Identify any SEO weaknesses or missing elements related to on-page SEO.
2. Suggest improvements that are specific to the content and keywords of the page (not generic).
3. Prioritize suggestions by impact (high, medium, low).
4. Output the suggestions as a JSON list with fields: type (content, technical, structure), priority, and suggestion text.

Example output:

[
  {{
    "type": "content",
    "priority": "high",
    "suggestion": "Include the primary keyword 'woodcraft router jigs' in the H1 heading to improve relevance."
  }},
  {{
    "type": "technical",
    "priority": "medium",
    "suggestion": "Compress images to improve page load speed from 75 to over 85."
  }},
  {{
    "type": "structure",
    "priority": "high",
    "suggestion": "Remove duplicate H1 tags; use only one per page for proper SEO hierarchy."
  }}
]

Remember, tailor your suggestions specifically for the page's content and keywords, avoiding generic advice.

Website Analysis Data:
URL: {url}
Title: {title}
Meta Description: {description}
Headings: {headings_text if headings_text else 'No headings found'}
Content Sample: {content_sample if content_sample else 'No content sample available'}
Top Keywords: {', '.join(keywords) if keywords else 'No keywords extracted'}
Page Speed Score: {seo_data.get('page_speed_score', 'Not available')}
Mobile Friendly: {seo_data.get('mobile_friendly', 'Not available')}

Analyze this website data and provide 5-8 specific SEO suggestions in JSON format. Focus on the actual content and structure, not generic advice. Remember to format your response as valid JSON only - no other text outside the JSON array."""

        return prompt

    def _validate_suggestions(self, suggestions: List[Dict]) -> List[Dict]:
        """Validate and clean AI-generated suggestions"""
        validated = []

        if not isinstance(suggestions, list):
            return []

        for suggestion in suggestions:
            if isinstance(suggestion, dict):
                # Ensure required fields exist
                cleaned_suggestion = {
                    'type': suggestion.get('type', 'General'),
                    'priority': suggestion.get('priority', 'Medium'),
                    'suggestion': suggestion.get('suggestion', 'No suggestion provided')
                }

                # Validate priority values
                if cleaned_suggestion['priority'] not in ['High', 'Medium', 'Low']:
                    cleaned_suggestion['priority'] = 'Medium'

                validated.append(cleaned_suggestion)

        return validated[:10]  # Limit to 10 suggestions

    def _fallback_suggestions(self, seo_data: Dict, keywords: List[str]) -> List[Dict]:
        """Fallback to rule-based suggestions when AI is unavailable"""
        suggestions = []

        # Ensure keywords is always a list
        if not isinstance(keywords, list):
            keywords = []

        # Title analysis
        title = seo_data.get('title', '')
        if not title:
            suggestions.append({
                'type': 'Meta',
                'priority': 'High',
                'suggestion': 'Add a descriptive title tag (50-60 characters) that includes your primary keyword.'
            })
        elif len(title) < 30:
            suggestions.append({
                'type': 'Meta',
                'priority': 'Medium',
                'suggestion': f'Title is too short ({len(title)} chars). Expand to 50-60 characters with keywords.'
            })
        elif len(title) > 60:
            suggestions.append({
                'type': 'Meta',
                'priority': 'Medium',
                'suggestion': f'Title is too long ({len(title)} chars). Shorten to under 60 characters.'
            })

        # Check if keywords are in title
        if keywords and title:
            primary_keyword = keywords[0].lower()
            if primary_keyword not in title.lower():
                suggestions.append({
                    'type': 'Meta',
                    'priority': 'High',
                    'suggestion': f'Include your primary keyword "{keywords[0]}" in the title tag.'
                })

        # Meta description analysis
        description = seo_data.get('description', '')
        if not description:
            suggestions.append({
                'type': 'Meta',
                'priority': 'High',
                'suggestion': 'Add a compelling meta description (150-160 characters) with your primary keyword.'
            })
        elif len(description) > 160:
            suggestions.append({
                'type': 'Meta',
                'priority': 'Medium',
                'suggestion': f'Meta description too long ({len(description)} chars). Shorten to under 160 characters.'
            })
        elif keywords and description:
            primary_keyword = keywords[0].lower()
            if primary_keyword not in description.lower():
                suggestions.append({
                    'type': 'Meta',
                    'priority': 'Medium',
                    'suggestion': f'Include your primary keyword "{keywords[0]}" in the meta description.'
                })

        # Headings analysis
        headings = seo_data.get('headings', {})
        h1_tags = headings.get('h1', [])
        
        if not h1_tags:
            suggestions.append({
                'type': 'Structure',
                'priority': 'High',
                'suggestion': 'Add an H1 heading that includes your primary keyword and describes the main topic.'
            })
        elif len(h1_tags) > 1:
            suggestions.append({
                'type': 'Structure',
                'priority': 'High',
                'suggestion': f'Multiple H1 tags found ({len(h1_tags)}). Use only one H1 per page.'
            })
        elif keywords and h1_tags:
            h1_text = ' '.join(h1_tags).lower()
            primary_keyword = keywords[0].lower()
            if primary_keyword not in h1_text:
                suggestions.append({
                    'type': 'Structure',
                    'priority': 'Medium',
                    'suggestion': f'Include your primary keyword "{keywords[0]}" in the H1 heading.'
                })

        # Check for H2 structure
        h2_tags = headings.get('h2', [])
        if len(h2_tags) == 0:
            suggestions.append({
                'type': 'Structure',
                'priority': 'Medium',
                'suggestion': 'Add H2 subheadings to structure your content and improve readability.'
            })

        # Content analysis
        content = seo_data.get('content_text', '')
        if not content:
            suggestions.append({
                'type': 'Content',
                'priority': 'High',
                'suggestion': 'Add substantial content (300+ words) relevant to your target keywords.'
            })
        elif len(content.split()) < 300:
            suggestions.append({
                'type': 'Content',
                'priority': 'Medium',
                'suggestion': f'Content is short ({len(content.split())} words). Aim for 300+ words for better SEO.'
            })

        # Keywords analysis
        if keywords and content:
            content_lower = content.lower()
            missing_keywords = []
            for keyword in keywords[:3]:
                if keyword.lower() not in content_lower:
                    missing_keywords.append(keyword)
            
            if missing_keywords:
                suggestions.append({
                    'type': 'Keywords',
                    'priority': 'High',
                    'suggestion': f'Include these keywords naturally in your content: {", ".join(missing_keywords)}'
                })

        # Image analysis
        images = seo_data.get('images', [])
        if images:
            missing_alt = sum(1 for img in images if not img.get('alt', '').strip())
            if missing_alt > 0:
                suggestions.append({
                    'type': 'Technical',
                    'priority': 'Medium',
                    'suggestion': f'{missing_alt} images missing alt text. Add descriptive alt text for accessibility.'
                })

        # Technical SEO
        page_speed = seo_data.get('page_speed_score', 0)
        if page_speed > 0 and page_speed < 70:
            suggestions.append({
                'type': 'Technical',
                'priority': 'High',
                'suggestion': f'Page speed is slow ({page_speed}/100). Optimize images and enable compression.'
            })

        if not seo_data.get('mobile_friendly', True):
            suggestions.append({
                'type': 'Technical',
                'priority': 'High',
                'suggestion': 'Page is not mobile-friendly. Implement responsive design.'
            })

        return suggestions[:8]  # Limit to 8 suggestions