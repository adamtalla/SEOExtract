from typing import List, Dict
from dataclasses import dataclass

@dataclass
class KeywordResult:
    """Represents a keyword extraction result with SEO metadata"""
    keyword: str
    score: float
    category: str  # 'service', 'location', 'commercial', 'technical', 'brand', 'general'
    frequency: int
    commercial_intent: bool

    def to_dict(self) -> Dict:
        return {
            'keyword': self.keyword,
            'score': round(self.score, 2),
            'category': self.category,
            'frequency': self.frequency,
            'commercial_intent': self.commercial_intent
        }

class EnhancedKeywordExtractor:
    """Enhanced keyword extractor that returns detailed results with classification"""

    def __init__(self):
        from ai_keyword_extractor import AIKeywordExtractor
        self.ai_extractor = AIKeywordExtractor()

    def extract_keywords_detailed(self, text: str, url: str = "", max_keywords: int = 10) -> List[KeywordResult]:
        """Extract keywords with detailed scoring and classification"""
        try:
            if not text or len(text.strip()) < 50:
                return []

            # Use the base AI extractor for initial extraction
            try:
                from ai_keyword_extractor import AIKeywordExtractor
                base_extractor = AIKeywordExtractor()
                keywords = base_extractor.extract_keywords(text, url, max_keywords * 2)
            except Exception as e:
                import logging
                logging.warning(f"AI extractor failed, using fallback: {e}")
                keywords = self._fallback_keyword_extraction(text, max_keywords * 2)

            results = []
            text_lower = text.lower()

            for keyword in keywords:
                if len(results) >= max_keywords:
                    break

                # Calculate detailed metrics
                frequency = text_lower.count(keyword.lower())
                commercial_intent = self._check_commercial_intent(keyword)
                category = self._classify_keyword_simple(keyword)
                score = self._calculate_detailed_score(keyword, text, url)

                if score > 0.5:  # Quality threshold
                    result = KeywordResult(
                        keyword=keyword,
                        score=score,
                        category=category,
                        frequency=frequency,
                        commercial_intent=commercial_intent
                    )
                    results.append(result)

            # Sort by score and return
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:max_keywords]

        except Exception as e:
            import logging
            logging.error(f"Error in detailed keyword extraction: {str(e)}")
            return []

    def _fallback_keyword_extraction(self, text: str, max_keywords: int) -> List[str]:
        """Simple fallback keyword extraction"""
        import re
        from collections import Counter

        # Clean text
        clean_text = re.sub(r'<[^>]+>', ' ', text)
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)

        # Extract words and phrases
        words = re.findall(r'\b[a-zA-Z]{3,}\b', clean_text.lower())

        # Basic stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

        # Filter and count
        filtered_words = [w for w in words if w not in stop_words and len(w) >= 3]
        word_counts = Counter(filtered_words)

        return [word for word, count in word_counts.most_common(max_keywords)]

    def _check_commercial_intent(self, keyword: str) -> bool:
        """Simple commercial intent check"""
        commercial_terms = ['buy', 'shop', 'store', 'service', 'professional', 'tools', 'equipment']
        keyword_lower = keyword.lower()
        return any(term in keyword_lower for term in commercial_terms)

    def _classify_keyword_simple(self, keyword: str) -> str:
        """Simple keyword classification"""
        keyword_lower = keyword.lower()

        if any(term in keyword_lower for term in ['service', 'tool', 'equipment', 'system']):
            return 'service'
        elif any(term in keyword_lower for term in ['professional', 'commercial', 'industrial']):
            return 'commercial'
        elif any(term in keyword_lower for term in ['wood', 'metal', 'craft', 'router', 'saw']):
            return 'technical'
        else:
            return 'general'

    def _calculate_detailed_score(self, keyword: str, text: str, url: str) -> float:
        """Calculate a detailed SEO importance score"""
        base_score = 1.0
        keyword_lower = keyword.lower()
        text_lower = text.lower()

        # Frequency scoring (but avoid keyword stuffing)
        frequency = text_lower.count(keyword_lower)
        text_length = len(text.split())
        frequency_ratio = frequency / text_length * 100

        if 0.5 <= frequency_ratio <= 2.0:
            base_score *= 1.5
        elif 2.0 < frequency_ratio <= 4.0:
            base_score *= 1.2
        elif frequency_ratio > 4.0:
            base_score *= 0.8  # Penalty for over-optimization

        # Position scoring - earlier is better
        first_occurrence = text_lower.find(keyword_lower)
        if first_occurrence >= 0:
            position_factor = max(0.7, 1.3 - (first_occurrence / len(text)))
            base_score *= position_factor

        # Length scoring - multi-word phrases are valuable
        word_count = len(keyword.split())
        if word_count == 2:
            base_score *= 1.4
        elif word_count == 3:
            base_score *= 1.6
        elif word_count >= 4:
            base_score *= 1.3

        # Commercial intent bonus
        if self.ai_extractor._has_commercial_intent(keyword):
            base_score *= 1.5

        # Technical term bonus
        if self.ai_extractor._is_technical_term(keyword):
            base_score *= 1.3

        # URL relevance bonus
        if url and keyword_lower in url.lower():
            base_score *= 1.2

        return base_score

    def get_keywords_by_category(self, results: List[KeywordResult]) -> Dict[str, List[str]]:
        """Group keywords by category"""
        categories = {}
        for result in results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result.keyword)
        return categories

    def get_commercial_keywords(self, results: List[KeywordResult]) -> List[str]:
        """Get keywords with commercial intent"""
        return [result.keyword for result in results if result.commercial_intent]