
import os
import json
import logging
import openai
from typing import Dict, List

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
    
    def generate_hybrid_suggestions(self, url: str, seo_data: Dict, keywords: List[str], user_plan: str = 'free') -> List[Dict]:
        """
        Generate suggestions based on user plan - AI for premium, rule-based for others
        """
        if user_plan == 'premium' and self.openai_api_key:
            return self.generate_ai_suggestions(url, seo_data, keywords)
        else:
            return self._fallback_suggestions(seo_data, keywords)
    
    def _build_analysis_prompt(self, url: str, seo_data: Dict, keywords: List[str]) -> str:
        """Build the analysis prompt with actual website data"""
        
        # Get content sample (first 500 chars)
        content_sample = seo_data.get('content_text', '')[:500] if seo_data.get('content_text') else ''
        
        # Format headings for display
        headings_text = ""
        headings = seo_data.get('headings', {})
        for level, heading_list in headings.items():
            if heading_list:
                headings_text += f"{level.upper()}: {', '.join(heading_list[:3])}; "
        
        prompt = f"""You are an expert SEO analyst.

You will be given real SEO data extracted from a website. Based on this data, provide smart, site-specific SEO improvement suggestions. Do not give generic advice — tailor your suggestions directly to what you observe.

--- Website Info ---
URL: {url}
Title: {seo_data.get("title", "No title found")}
Meta Description: {seo_data.get("description", "No meta description found")}
Headings: {headings_text or "No headings found"}
Content Sample: {content_sample or "No content sample available"}
Top Keywords Found: {', '.join(keywords) if keywords else "No keywords extracted"}

--- Instructions ---
Analyze this website and give 5–10 high-impact SEO recommendations tailored to the actual content and structure. Use this format for each:

- Type: (Meta, Content, Keywords, Structure, Speed, Mobile, Schema)
- Priority: (High, Medium, Low)
- Suggestion: One specific and practical suggestion for improving SEO based on the data.

Return the output as a JSON array of objects like:
[
  {{
    "type": "Content",
    "priority": "High",
    "suggestion": "Add more keyword-rich content related to 'router jigs' since it's a top keyword but appears very little in the content."
  }}
]"""
        
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
        
        # Headings analysis
        headings = seo_data.get('headings', {})
        h1_tags = headings.get('h1', [])
        if not h1_tags:
            suggestions.append({
                'type': 'Structure',
                'priority': 'High',
                'suggestion': 'Add a clear H1 heading that includes your primary keyword.'
            })
        elif len(h1_tags) > 1:
            suggestions.append({
                'type': 'Structure',
                'priority': 'Medium',
                'suggestion': f'Multiple H1 tags found ({len(h1_tags)}). Use only one H1 per page.'
            })
        
        # Keywords analysis
        if keywords:
            content_text = seo_data.get('content_text', '').lower()
            for keyword in keywords[:3]:  # Check top 3 keywords
                if keyword.lower() not in content_text:
                    suggestions.append({
                        'type': 'Content',
                        'priority': 'Medium',
                        'suggestion': f'Include the keyword "{keyword}" naturally in your content for better relevance.'
                    })
        
        # Mobile friendliness
        if not seo_data.get('mobile_friendly', True):
            suggestions.append({
                'type': 'Mobile',
                'priority': 'High',
                'suggestion': 'Add viewport meta tag for mobile-friendly display.'
            })
        
        return suggestions[:8]  # Return up to 8 suggestions
