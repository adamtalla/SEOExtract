
import re
import logging
from typing import List, Dict, Tuple
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.stem import WordNetLemmatizer
import numpy as np

# Download required NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/words')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logging.info("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('wordnet', quiet=True)

class AIKeywordExtractor:
    """AI-powered keyword extraction using NLP techniques and semantic analysis"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Extend stop words with web-specific terms
        self.stop_words.update({
            'website', 'page', 'site', 'web', 'home', 'main', 'content', 'click', 'here',
            'read', 'more', 'contact', 'about', 'privacy', 'terms', 'cookie', 'login',
            'register', 'signup', 'user', 'account', 'profile', 'settings', 'help',
            'support', 'faq', 'blog', 'news', 'article', 'post', 'comment', 'share',
            'social', 'media', 'follow', 'subscribe', 'newsletter', 'email', 'phone',
            'address', 'location', 'map', 'search', 'filter', 'sort', 'view', 'show',
            'hide', 'menu', 'navigation', 'nav', 'header', 'footer', 'sidebar',
            'widget', 'button', 'link', 'image', 'video', 'audio', 'download',
            'upload', 'file', 'document', 'pdf', 'jpg', 'png', 'gif', 'svg'
        })
        
        # Industry-specific keyword patterns
        self.keyword_patterns = {
            'tech': ['software', 'app', 'platform', 'system', 'tool', 'solution', 'technology'],
            'business': ['service', 'company', 'business', 'enterprise', 'corporate', 'professional'],
            'ecommerce': ['product', 'price', 'buy', 'purchase', 'order', 'shop', 'store'],
            'education': ['course', 'training', 'learn', 'education', 'tutorial', 'guide'],
            'health': ['health', 'medical', 'doctor', 'treatment', 'care', 'wellness'],
            'finance': ['finance', 'money', 'investment', 'bank', 'loan', 'insurance']
        }
    
    def extract_keywords(self, text: str, url: str = "", max_keywords: int = 10) -> List[str]:
        """Extract keywords using AI-powered semantic analysis"""
        try:
            if not text or len(text.strip()) < 50:
                logging.warning("Text too short for meaningful keyword extraction")
                return []
            
            # Preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Extract different types of keywords
            semantic_keywords = self._extract_semantic_keywords(cleaned_text)
            entity_keywords = self._extract_named_entities(cleaned_text)
            phrase_keywords = self._extract_key_phrases(cleaned_text)
            domain_keywords = self._extract_domain_keywords(text, url)
            
            # Combine and score all keywords
            all_keywords = {}
            
            # Add semantic keywords with high weight
            for kw, score in semantic_keywords.items():
                all_keywords[kw] = score * 1.5
            
            # Add entity keywords with medium-high weight
            for kw, score in entity_keywords.items():
                all_keywords[kw] = all_keywords.get(kw, 0) + score * 1.3
            
            # Add phrase keywords with medium weight
            for kw, score in phrase_keywords.items():
                all_keywords[kw] = all_keywords.get(kw, 0) + score * 1.0
            
            # Add domain keywords with boost
            for kw, score in domain_keywords.items():
                all_keywords[kw] = all_keywords.get(kw, 0) + score * 1.2
            
            # Sort by score and return top keywords
            sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)
            
            # Filter and clean results
            final_keywords = []
            seen = set()
            
            for keyword, score in sorted_keywords:
                if len(final_keywords) >= max_keywords:
                    break
                
                keyword_clean = keyword.lower().strip()
                if (keyword_clean not in seen and 
                    self._is_quality_keyword(keyword) and
                    len(keyword_clean) >= 3):
                    final_keywords.append(keyword)
                    seen.add(keyword_clean)
            
            return final_keywords
            
        except Exception as e:
            logging.error(f"Error in AI keyword extraction: {str(e)}")
            return self._fallback_extraction(text, max_keywords)
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        # Remove HTML tags and special characters
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short and very long words
        words = text.split()
        filtered_words = [w for w in words if 2 <= len(w) <= 25]
        
        return ' '.join(filtered_words)
    
    def _extract_semantic_keywords(self, text: str) -> Dict[str, float]:
        """Extract semantically important keywords using POS tagging and frequency"""
        keywords = {}
        
        # Tokenize and tag parts of speech
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        # Focus on nouns, adjectives, and meaningful verbs
        important_pos = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VB', 'VBG', 'VBN'}
        
        # Extract single words
        for word, pos in pos_tags:
            if (pos in important_pos and 
                word not in self.stop_words and 
                len(word) >= 3 and
                not word.isdigit()):
                
                lemmatized = self.lemmatizer.lemmatize(word)
                keywords[lemmatized] = keywords.get(lemmatized, 0) + self._calculate_word_importance(word, pos)
        
        # Extract noun phrases (consecutive nouns/adjectives)
        noun_phrases = self._extract_noun_phrases(pos_tags)
        for phrase in noun_phrases:
            if len(phrase.split()) <= 3:  # Limit phrase length
                keywords[phrase] = keywords.get(phrase, 0) + 2.0
        
        return keywords
    
    def _extract_named_entities(self, text: str) -> Dict[str, float]:
        """Extract named entities (persons, organizations, locations)"""
        keywords = {}
        
        try:
            # Tokenize and extract named entities
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            entities = ne_chunk(pos_tags)
            
            for chunk in entities:
                if hasattr(chunk, 'label'):
                    entity_name = ' '.join([token for token, pos in chunk.leaves()])
                    entity_type = chunk.label()
                    
                    # Weight different entity types
                    weights = {
                        'ORGANIZATION': 2.0,
                        'GPE': 1.5,  # Geopolitical entities
                        'PERSON': 1.3,
                        'MONEY': 1.2,
                        'PERCENT': 1.2
                    }
                    
                    weight = weights.get(entity_type, 1.0)
                    keywords[entity_name.lower()] = keywords.get(entity_name.lower(), 0) + weight
                    
        except Exception as e:
            logging.warning(f"Named entity extraction failed: {str(e)}")
        
        return keywords
    
    def _extract_key_phrases(self, text: str) -> Dict[str, float]:
        """Extract meaningful phrases using collocation and co-occurrence"""
        keywords = {}
        
        # Split into sentences for better phrase extraction
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            
            # Extract 2-3 word phrases
            for i in range(len(words) - 1):
                # Bigrams
                bigram = f"{words[i]} {words[i+1]}"
                if self._is_meaningful_phrase(bigram):
                    keywords[bigram] = keywords.get(bigram, 0) + 1.0
                
                # Trigrams
                if i < len(words) - 2:
                    trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                    if self._is_meaningful_phrase(trigram):
                        keywords[trigram] = keywords.get(trigram, 0) + 1.5
        
        return keywords
    
    def _extract_domain_keywords(self, text: str, url: str) -> Dict[str, float]:
        """Extract domain-specific keywords based on content analysis"""
        keywords = {}
        text_lower = text.lower()
        
        # Detect industry/domain
        domain_scores = {}
        for domain, patterns in self.keyword_patterns.items():
            score = sum(text_lower.count(pattern) for pattern in patterns)
            domain_scores[domain] = score
        
        # Get the most likely domain
        primary_domain = max(domain_scores, key=domain_scores.get) if domain_scores else None
        
        # Extract keywords specific to the detected domain
        if primary_domain and domain_scores[primary_domain] > 0:
            domain_patterns = self.keyword_patterns[primary_domain]
            for pattern in domain_patterns:
                if pattern in text_lower:
                    keywords[pattern] = 2.0
        
        # Extract URL-based keywords
        if url:
            url_keywords = self._extract_url_keywords(url)
            for kw in url_keywords:
                keywords[kw] = keywords.get(kw, 0) + 1.5
        
        return keywords
    
    def _extract_url_keywords(self, url: str) -> List[str]:
        """Extract meaningful keywords from URL structure"""
        keywords = []
        
        try:
            # Extract from path segments
            path_segments = url.split('/')
            for segment in path_segments:
                # Clean segment
                segment = re.sub(r'[^\w\-]', ' ', segment)
                words = segment.split()
                
                for word in words:
                    if (len(word) >= 3 and 
                        word.lower() not in self.stop_words and
                        not word.isdigit()):
                        keywords.append(word.lower())
        
        except Exception as e:
            logging.warning(f"URL keyword extraction failed: {str(e)}")
        
        return keywords
    
    def _extract_noun_phrases(self, pos_tags: List[Tuple[str, str]]) -> List[str]:
        """Extract noun phrases from POS tagged text"""
        phrases = []
        current_phrase = []
        
        # Pattern: (Adjective)* (Noun)+
        for word, pos in pos_tags:
            if pos.startswith('JJ') or pos.startswith('NN'):  # Adjective or Noun
                if word.lower() not in self.stop_words and len(word) >= 3:
                    current_phrase.append(word.lower())
            else:
                if len(current_phrase) >= 2:  # At least 2 words
                    phrases.append(' '.join(current_phrase))
                current_phrase = []
        
        # Handle last phrase
        if len(current_phrase) >= 2:
            phrases.append(' '.join(current_phrase))
        
        return phrases
    
    def _calculate_word_importance(self, word: str, pos: str) -> float:
        """Calculate importance score for a word based on POS and characteristics"""
        base_score = 1.0
        
        # POS-based scoring
        pos_weights = {
            'NN': 1.5, 'NNS': 1.5, 'NNP': 2.0, 'NNPS': 2.0,  # Nouns
            'JJ': 1.2, 'JJR': 1.2, 'JJS': 1.2,  # Adjectives
            'VB': 0.8, 'VBG': 1.0, 'VBN': 1.0   # Verbs
        }
        
        base_score *= pos_weights.get(pos, 1.0)
        
        # Length-based scoring (prefer medium-length words)
        word_len = len(word)
        if 4 <= word_len <= 8:
            base_score *= 1.2
        elif word_len >= 12:
            base_score *= 0.8
        
        # Capitalization bonus (might be proper nouns)
        if word[0].isupper():
            base_score *= 1.1
        
        return base_score
    
    def _is_meaningful_phrase(self, phrase: str) -> bool:
        """Check if a phrase is meaningful for keyword extraction"""
        words = phrase.split()
        
        # Filter out phrases with too many stop words
        stop_word_count = sum(1 for word in words if word in self.stop_words)
        if stop_word_count > len(words) / 2:
            return False
        
        # Check minimum length
        if len(phrase) < 5:
            return False
        
        # Avoid purely functional phrases
        functional_patterns = [
            r'^(click|read|see|view|show)',
            r'(here|there|now|then)$',
            r'^(the|a|an)\s',
            r'\d+\s*(th|st|nd|rd)'
        ]
        
        for pattern in functional_patterns:
            if re.search(pattern, phrase, re.IGNORECASE):
                return False
        
        return True
    
    def _is_quality_keyword(self, keyword: str) -> bool:
        """Determine if a keyword meets quality standards"""
        keyword_lower = keyword.lower().strip()
        
        # Basic filters
        if (keyword_lower in self.stop_words or
            len(keyword_lower) < 3 or
            keyword_lower.isdigit() or
            not re.match(r'^[a-zA-Z0-9\s\-]+$', keyword)):
            return False
        
        # Avoid overly generic terms
        generic_terms = {
            'content', 'information', 'details', 'data', 'things', 'stuff',
            'items', 'example', 'sample', 'text', 'words', 'title'
        }
        
        if keyword_lower in generic_terms:
            return False
        
        # Check for repeated characters (likely errors)
        if re.search(r'(.)\1{3,}', keyword_lower):
            return False
        
        return True
    
    def _fallback_extraction(self, text: str, max_keywords: int) -> List[str]:
        """Fallback keyword extraction method"""
        try:
            # Simple frequency-based extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            word_freq = Counter(words)
            
            # Filter out stop words
            filtered_words = {
                word: freq for word, freq in word_freq.items()
                if word not in self.stop_words and len(word) >= 3
            }
            
            # Return top keywords
            return [word for word, freq in 
                   sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:max_keywords]]
        
        except Exception as e:
            logging.error(f"Fallback extraction failed: {str(e)}")
            return []

# Test function
def test_ai_extraction():
    """Test the AI keyword extractor"""
    extractor = AIKeywordExtractor()
    
    test_text = """
    Welcome to TechSolutions, a leading software development company specializing in 
    artificial intelligence and machine learning solutions. We provide cutting-edge 
    cloud computing services, data analytics platforms, and custom software development 
    for enterprise clients. Our team of experienced developers and data scientists 
    work with Python, JavaScript, and modern frameworks to deliver scalable solutions.
    """
    
    keywords = extractor.extract_keywords(test_text, "https://techsolutions.com/services")
    print(f"Extracted keywords: {keywords}")
    return keywords

if __name__ == "__main__":
    test_ai_extraction()
