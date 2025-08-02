
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

# Import the training system
try:
    from seo_keyword_trainer import get_keyword_trainer
    TRAINER_AVAILABLE = True
except ImportError:
    TRAINER_AVAILABLE = False
    logging.warning("SEO keyword trainer not available")

# Download required NLTK data if not present
required_nltk_data = [
    ('tokenizers/punkt', 'punkt'),
    ('tokenizers/punkt_tab', 'punkt_tab'),
    ('corpora/stopwords', 'stopwords'),
    ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
    ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
    ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
    ('corpora/words', 'words'),
    ('corpora/wordnet', 'wordnet')
]

for resource_path, download_name in required_nltk_data:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        logging.info(f"Downloading NLTK resource: {download_name}")
        try:
            nltk.download(download_name, quiet=True)
        except Exception as e:
            logging.warning(f"Failed to download {download_name}: {e}")

class AIKeywordExtractor:
    """AI-powered keyword extraction using NLP techniques and semantic analysis"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Extend stop words with web-specific and SEO-irrelevant terms
        self.stop_words.update({
            # Generic web terms
            'website', 'page', 'site', 'web', 'home', 'main', 'content', 'click', 'here',
            'read', 'more', 'contact', 'about', 'privacy', 'terms', 'cookie', 'login',
            'register', 'signup', 'user', 'account', 'profile', 'settings', 'help',
            'support', 'faq', 'blog', 'news', 'article', 'post', 'comment', 'share',
            'social', 'media', 'follow', 'subscribe', 'newsletter', 'email', 'phone',
            'address', 'location', 'map', 'search', 'filter', 'sort', 'view', 'show',
            'hide', 'menu', 'navigation', 'nav', 'header', 'footer', 'sidebar',
            'widget', 'button', 'link', 'image', 'video', 'audio', 'download',
            'upload', 'file', 'document', 'pdf', 'jpg', 'png', 'gif', 'svg',
            # Action/filler words that aren't SEO valuable
            'call', 'offer', 'today', 'now', 'get', 'find', 'learn', 'discover',
            'explore', 'browse', 'visit', 'try', 'start', 'begin', 'continue',
            'join', 'become', 'make', 'take', 'give', 'receive', 'send', 'buy',
            'order', 'purchase', 'choose', 'select', 'pick', 'decide', 'want',
            'need', 'use', 'enjoy', 'love', 'like', 'prefer', 'recommend',
            # Generic descriptors
            'best', 'great', 'good', 'excellent', 'amazing', 'awesome', 'perfect',
            'ideal', 'ultimate', 'complete', 'full', 'total', 'entire', 'whole',
            'all', 'every', 'each', 'any', 'some', 'many', 'most', 'few', 'several',
            'various', 'different', 'multiple', 'numerous', 'countless', 'unlimited',
            # Time/date terms
            'year', 'month', 'week', 'day', 'hour', 'minute', 'second', 'time',
            'date', 'schedule', 'calendar', 'appointment', 'meeting', 'event',
            # Generic nouns that add no SEO value
            'thing', 'stuff', 'item', 'object', 'element', 'part', 'piece',
            'section', 'area', 'place', 'spot', 'point', 'way', 'method',
            'approach', 'solution', 'option', 'choice', 'alternative', 'possibility'
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
        """Extract high-quality SEO keywords using enhanced analysis"""
        try:
            if not text or len(text.strip()) < 50:
                logging.warning("Text too short for meaningful keyword extraction")
                return []
            
            # Extract HTML structure for better phrase detection
            heading_keywords = self._extract_heading_keywords(text)
            product_keywords = self._extract_product_service_keywords(text)
            noun_phrase_keywords = self._extract_quality_noun_phrases(text)
            domain_keywords = self._extract_domain_keywords(text, url)
            
            # Combine all keywords with weighted scoring
            all_keywords = {}
            
            # Heading keywords get highest priority (H1, H2, etc.)
            for kw, score in heading_keywords.items():
                all_keywords[kw] = score * 3.0
            
            # Product/service keywords get high priority
            for kw, score in product_keywords.items():
                all_keywords[kw] = all_keywords.get(kw, 0) + score * 2.5
            
            # Quality noun phrases get medium-high priority
            for kw, score in noun_phrase_keywords.items():
                all_keywords[kw] = all_keywords.get(kw, 0) + score * 2.0
            
            # Domain-specific keywords get medium priority
            for kw, score in domain_keywords.items():
                all_keywords[kw] = all_keywords.get(kw, 0) + score * 1.5
            
            # Apply final scoring and filtering
            scored_keywords = self._apply_final_scoring(all_keywords, text, url)
            
            # Sort by score and apply quality filters
            final_keywords = []
            seen = set()
            
            # Get all candidates first
            candidates = []
            for keyword, score in sorted(scored_keywords.items(), key=lambda x: x[1], reverse=True):
                keyword_clean = keyword.lower().strip()
                if (keyword_clean not in seen and 
                    self._is_high_quality_keyword(keyword) and
                    not self._is_weak_fragment(keyword)):
                    candidates.append(keyword)
                    seen.add(keyword_clean)
            
            # Apply enhanced training-based filtering if available
            if TRAINER_AVAILABLE:
                try:
                    trainer = get_keyword_trainer()
                    if trainer.is_trained:
                        # Step 1: Direct filtering to remove bad keywords (fast)
                        direct_filtered = trainer.filter_keywords_direct(candidates, remove_bad=True, use_fuzzy=True)
                        
                        # Step 2: Score and sort remaining keywords
                        scored_keywords = trainer.score_and_sort_keywords(direct_filtered, use_fuzzy=True)
                        
                        # Step 3: Take top keywords based on combined scoring
                        final_keywords = [kw for kw, score in scored_keywords[:max_keywords]]
                        
                        # Log detailed filtering stats
                        stats = trainer.get_training_stats()
                        logging.info(f"Applied direct + AI filtering with {stats['total_examples']} examples:")
                        logging.info(f"  Initial candidates: {len(candidates)}")
                        logging.info(f"  After direct filtering: {len(direct_filtered)}")
                        logging.info(f"  Final keywords: {len(final_keywords)}")
                        logging.info(f"  Training data: {stats['good_examples']} good, {stats['bad_examples']} bad")
                    else:
                        final_keywords = candidates[:max_keywords]
                        logging.warning("Trainer not properly trained, using fallback")
                except Exception as e:
                    logging.warning(f"Training-based filtering failed: {e}")
                    final_keywords = candidates[:max_keywords]
            else:
                final_keywords = candidates[:max_keywords]
                logging.warning("Trainer not available, using base filtering")
            
            return final_keywords
            
        except Exception as e:
            logging.error(f"Error in AI keyword extraction: {str(e)}")
            return self._enhanced_fallback_extraction(text, max_keywords)
    
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
    
    def _extract_seo_focused_keywords(self, text: str, url: str) -> Dict[str, float]:
        """Extract keywords specifically valuable for SEO"""
        keywords = {}
        text_lower = text.lower()
        
        # Business/service indicators
        service_patterns = [
            r'\b(\w+)\s+services?\b', r'\b(\w+)\s+solutions?\b', r'\b(\w+)\s+company\b',
            r'\b(\w+)\s+specialist\b', r'\b(\w+)\s+expert\b', r'\b(\w+)\s+professional\b',
            r'\b(\w+)\s+contractor\b', r'\b(\w+)\s+repair\b', r'\b(\w+)\s+installation\b',
            r'\b(\w+)\s+maintenance\b', r'\b(\w+)\s+cleaning\b', r'\b(\w+)\s+consulting\b'
        ]
        
        for pattern in service_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                service_term = match.group(1)
                if service_term not in self.stop_words and len(service_term) >= 3:
                    full_phrase = match.group(0)
                    keywords[full_phrase] = 3.0
                    keywords[service_term] = 2.0
        
        # Location-based keywords
        location_patterns = [
            r'\b(\w+)\s+(plumber|electrician|contractor|company|services?)\b',
            r'\b(in|near|around)\s+(\w+(?:\s+\w+)?)\b'
        ]
        
        for pattern in location_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                if len(match.groups()) >= 2:
                    location_term = match.group(1) if 'in|near|around' not in match.group(1) else match.group(2)
                    if location_term not in self.stop_words and len(location_term) >= 3:
                        keywords[match.group(0)] = 2.5
        
        # Emergency/urgent service keywords
        urgent_patterns = [
            r'\bemergency\s+(\w+)', r'\b24\s*hours?\s+(\w+)', r'\bimmediate\s+(\w+)',
            r'\bsame\s+day\s+(\w+)', r'\bfast\s+(\w+)', r'\bquick\s+(\w+)'
        ]
        
        for pattern in urgent_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                keywords[match.group(0)] = 2.8
        
        return keywords
    
    def _calculate_seo_importance(self, keywords: Dict[str, float], text: str, url: str) -> Dict[str, float]:
        """Calculate SEO importance scores based on multiple factors"""
        scored_keywords = {}
        text_lower = text.lower()
        text_length = len(text.split())
        
        for keyword, base_score in keywords.items():
            seo_score = base_score
            keyword_lower = keyword.lower()
            
            # Frequency factor (but not too high to avoid keyword stuffing)
            frequency = text_lower.count(keyword_lower)
            if frequency > 0:
                # Optimal frequency is 1-3% of content
                frequency_ratio = frequency / text_length * 100
                if 0.5 <= frequency_ratio <= 3.0:
                    seo_score *= 1.5
                elif frequency_ratio > 5.0:
                    seo_score *= 0.7  # Penalize potential keyword stuffing
            
            # Position factor - keywords near beginning are more valuable
            first_occurrence = text_lower.find(keyword_lower)
            if first_occurrence >= 0:
                position_factor = max(0.8, 1.2 - (first_occurrence / len(text)))
                seo_score *= position_factor
            
            # Length factor - multi-word phrases are often more valuable
            word_count = len(keyword.split())
            if word_count >= 2:
                seo_score *= 1.3
            if word_count >= 3:
                seo_score *= 1.2
            
            # Commercial intent factor
            if self._has_commercial_intent(keyword):
                seo_score *= 1.6
            
            # Technical/specific terms get bonus
            if self._is_technical_term(keyword):
                seo_score *= 1.4
            
            scored_keywords[keyword] = seo_score
        
        return scored_keywords
    
    def _is_personal_name(self, text: str) -> bool:
        """Check if text appears to be a personal name"""
        words = text.split()
        if len(words) > 3:  # Names are usually 1-3 words
            return False
        
        # Common name patterns
        name_patterns = [
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',  # First Last
            r'^[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+$',  # First M. Last
            r'^Dr\.\s+[A-Z][a-z]+',  # Dr. Name
            r'^Mr\.\s+[A-Z][a-z]+',  # Mr. Name
            r'^Mrs\.\s+[A-Z][a-z]+',  # Mrs. Name
        ]
        
        for pattern in name_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def _is_seo_valuable_keyword(self, keyword: str) -> bool:
        """Enhanced quality check for SEO value"""
        keyword_lower = keyword.lower().strip()
        
        # Basic quality checks
        if not self._is_quality_keyword(keyword):
            return False
        
        # Check for business/service relevance
        valuable_patterns = [
            r'\b\w+\s+(service|solution|company|repair|cleaning|installation)\b',
            r'\bemergency\s+\w+', r'\b24\s*hour', r'\bprofessional\s+\w+',
            r'\bcertified\s+\w+', r'\blicensed\s+\w+', r'\bcommercial\s+\w+',
            r'\bresidential\s+\w+', r'\baffordable\s+\w+', r'\bexperienced\s+\w+'
        ]
        
        for pattern in valuable_patterns:
            if re.search(pattern, keyword_lower):
                return True
        
        # Industry-specific terms are valuable
        if any(industry_term in keyword_lower for industry_terms in self.keyword_patterns.values() 
               for industry_term in industry_terms):
            return True
        
        # Multi-word phrases are generally more valuable
        if len(keyword.split()) >= 2:
            return True
        
        # Single technical/specific terms
        if len(keyword) >= 5 and not self._is_generic_filler(keyword):
            return True
        
        return False
    
    def _is_generic_filler(self, keyword: str) -> bool:
        """Check if keyword is generic filler without SEO value"""
        keyword_lower = keyword.lower().strip()
        
        # Generic action words
        generic_actions = {
            'call', 'contact', 'visit', 'browse', 'explore', 'discover', 'learn',
            'find', 'search', 'look', 'see', 'view', 'check', 'try', 'test',
            'start', 'begin', 'continue', 'finish', 'complete', 'end', 'stop'
        }
        
        # Generic descriptors
        generic_descriptors = {
            'best', 'great', 'good', 'excellent', 'amazing', 'awesome', 'perfect',
            'wonderful', 'fantastic', 'outstanding', 'superior', 'premium', 'quality',
            'top', 'leading', 'premier', 'ultimate', 'complete', 'full', 'total'
        }
        
        # Time-related terms
        time_terms = {
            'today', 'now', 'soon', 'later', 'tomorrow', 'yesterday', 'recently',
            'currently', 'presently', 'immediately', 'instantly', 'quickly', 'fast'
        }
        
        all_generic = generic_actions | generic_descriptors | time_terms
        
        return keyword_lower in all_generic
    
    def _has_commercial_intent(self, keyword: str) -> bool:
        """Check if keyword indicates commercial/buying intent"""
        commercial_indicators = [
            'buy', 'purchase', 'order', 'hire', 'book', 'schedule', 'appointment',
            'quote', 'estimate', 'price', 'cost', 'affordable', 'cheap', 'discount',
            'deal', 'offer', 'special', 'promotion', 'service', 'repair', 'fix',
            'install', 'replace', 'upgrade', 'maintenance', 'emergency', 'near me'
        ]
        
        keyword_lower = keyword.lower()
        return any(indicator in keyword_lower for indicator in commercial_indicators)
    
    def _is_technical_term(self, keyword: str) -> bool:
        """Check if keyword is a technical/specific industry term"""
        # Technical terms are usually:
        # - Longer than 5 characters
        # - Not in common dictionaries
        # - Industry-specific
        keyword_lower = keyword.lower()
        
        if len(keyword_lower) < 5:
            return False
        
        # Check if it contains technical patterns
        technical_patterns = [
            r'\w+ing$',  # -ing endings (plumbing, roofing)
            r'\w+tion$',  # -tion endings (installation, renovation)
            r'\w+ment$',  # -ment endings (treatment, equipment)
            r'\w+ance$',  # -ance endings (maintenance, insurance)
            r'\w+ical$',  # -ical endings (electrical, mechanical)
        ]
        
        for pattern in technical_patterns:
            if re.search(pattern, keyword_lower):
                return True
        
        return False
    
    def classify_keyword(self, keyword: str, context: str = "") -> str:
        """Classify keyword into SEO categories"""
        keyword_lower = keyword.lower()
        
        # Service/Product terms
        if any(term in keyword_lower for term in ['service', 'product', 'solution', 'repair', 'installation', 'maintenance']):
            return 'service'
        
        # Location terms
        if any(term in keyword_lower for term in ['near', 'in', 'location', 'local', 'area']) or self._is_location_term(keyword):
            return 'location'
        
        # Action/Commercial terms
        if self._has_commercial_intent(keyword):
            return 'commercial'
        
        # Technical terms
        if self._is_technical_term(keyword):
            return 'technical'
        
        # Brand/Company terms
        if any(term in keyword_lower for term in ['company', 'corp', 'inc', 'llc', 'business']):
            return 'brand'
        
        return 'general'
    
    def _is_location_term(self, keyword: str) -> bool:
        """Check if keyword is likely a location"""
        # This is a simplified check - in production you might use a location database
        location_indicators = [
            r'\b[A-Z][a-z]+\s+(city|town|county|state|street|avenue|road|drive)\b',
            r'\b(north|south|east|west|downtown|uptown)\s+\w+',
            r'\b\w+\s+(NY|NYC|CA|FL|TX|IL)\b'  # State abbreviations
        ]
        
        for pattern in location_indicators:
            if re.search(pattern, keyword, re.IGNORECASE):
                return True
        
        return False
    
    def _extract_heading_keywords(self, text: str) -> Dict[str, float]:
        """Extract keywords from HTML headings and important elements"""
        keywords = {}
        
        # Extract from HTML headings
        heading_patterns = [
            r'<h[1-6][^>]*>(.*?)</h[1-6]>',  # H1-H6 tags
            r'<title[^>]*>(.*?)</title>',     # Title tag
            r'<strong[^>]*>(.*?)</strong>',   # Strong/bold text
            r'<b[^>]*>(.*?)</b>',             # Bold text
            r'<em[^>]*>(.*?)</em>',           # Emphasized text
        ]
        
        for pattern in heading_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                heading_text = re.sub(r'<[^>]+>', '', match.group(1)).strip()
                if heading_text and len(heading_text) > 3:
                    # Extract meaningful phrases from headings
                    phrases = self._extract_phrases_from_text(heading_text)
                    for phrase in phrases:
                        if self._is_meaningful_heading_phrase(phrase):
                            keywords[phrase.lower()] = 4.0
        
        return keywords
    
    def _extract_product_service_keywords(self, text: str) -> Dict[str, float]:
        """Extract product and service-related keywords"""
        keywords = {}
        text_lower = text.lower()
        
        # Product/service patterns with higher specificity
        product_patterns = [
            r'\b(\w+(?:\s+\w+){0,2})\s+(?:tools?|equipment|supplies?|products?)\b',
            r'\b(\w+(?:\s+\w+){0,2})\s+(?:systems?|solutions?|services?)\b',
            r'\b(\w+(?:\s+\w+){0,2})\s+(?:machines?|devices?|instruments?)\b',
            r'\b(\w+(?:\s+\w+){0,2})\s+(?:kits?|sets?|collections?)\b',
            r'\b(?:professional|commercial|industrial)\s+(\w+(?:\s+\w+){0,2})\b',
            r'\b(\w+)\s+(?:saw|drill|router|lathe|planer|sander)\b',
            r'\b(\w+)\s+(?:collection|storage|organization)\b'
        ]
        
        for pattern in product_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                term = match.group(1).strip()
                if (len(term) >= 3 and 
                    term not in self.stop_words and 
                    not self._is_generic_filler(term)):
                    
                    full_phrase = match.group(0).strip()
                    keywords[full_phrase] = 3.5
                    if len(term.split()) >= 2:
                        keywords[term] = 2.5
        
        return keywords
    
    def _extract_quality_noun_phrases(self, text: str) -> Dict[str, float]:
        """Extract high-quality noun phrases using advanced NLP"""
        keywords = {}
        
        try:
            # Clean text for better processing
            clean_text = re.sub(r'<[^>]+>', ' ', text)
            clean_text = re.sub(r'[^\w\s\-]', ' ', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text)
            
            # Tokenize and POS tag
            tokens = word_tokenize(clean_text.lower())
            pos_tags = pos_tag(tokens)
            
            # Extract noun phrases with specific patterns
            current_phrase = []
            
            for i, (word, pos) in enumerate(pos_tags):
                # Include nouns, adjectives, and some verbs in gerund form
                if (pos.startswith(('NN', 'JJ')) or 
                    (pos == 'VBG' and word.endswith('ing')) or
                    word in ['dust', 'power', 'hand', 'wood', 'metal']):
                    
                    if (word not in self.stop_words and 
                        len(word) >= 3 and 
                        not word.isdigit()):
                        current_phrase.append(word)
                else:
                    # End of phrase - evaluate it
                    if len(current_phrase) >= 2:
                        phrase = ' '.join(current_phrase)
                        if self._is_valuable_noun_phrase(phrase):
                            keywords[phrase] = len(current_phrase) * 1.5
                    current_phrase = []
            
            # Handle final phrase
            if len(current_phrase) >= 2:
                phrase = ' '.join(current_phrase)
                if self._is_valuable_noun_phrase(phrase):
                    keywords[phrase] = len(current_phrase) * 1.5
        
        except Exception as e:
            logging.warning(f"Error in noun phrase extraction: {e}")
        
        return keywords
    
    def _extract_phrases_from_text(self, text: str) -> List[str]:
        """Extract meaningful phrases from text"""
        phrases = []
        
        # Split by common delimiters
        parts = re.split(r'[,;:\-\|]+', text)
        
        for part in parts:
            part = part.strip()
            if len(part) >= 5 and not self._is_generic_filler(part):
                # Extract 2-4 word phrases
                words = part.split()
                for i in range(len(words)):
                    for j in range(i + 2, min(i + 5, len(words) + 1)):
                        phrase = ' '.join(words[i:j])
                        if self._is_meaningful_phrase(phrase):
                            phrases.append(phrase)
        
        return phrases
    
    def _is_meaningful_heading_phrase(self, phrase: str) -> bool:
        """Check if a phrase from headings is meaningful"""
        phrase_lower = phrase.lower().strip()
        
        # Must be substantial
        if len(phrase_lower) < 5 or len(phrase_lower.split()) < 2:
            return False
        
        # Avoid generic heading text
        generic_headings = {
            'welcome to', 'about us', 'contact us', 'our services', 'our products',
            'home page', 'main menu', 'click here', 'learn more', 'get started',
            'sign up', 'log in', 'follow us', 'subscribe now'
        }
        
        for generic in generic_headings:
            if generic in phrase_lower:
                return False
        
        # Should contain meaningful terms
        valuable_indicators = [
            'tool', 'equipment', 'service', 'product', 'system', 'collection',
            'professional', 'commercial', 'industrial', 'woodworking', 'metal',
            'craft', 'workshop', 'shop', 'store', 'supply', 'manufacturer'
        ]
        
        return any(indicator in phrase_lower for indicator in valuable_indicators)
    
    def _is_valuable_noun_phrase(self, phrase: str) -> bool:
        """Check if a noun phrase is valuable for SEO"""
        phrase_lower = phrase.lower().strip()
        words = phrase_lower.split()
        
        # Basic quality checks
        if len(words) < 2 or len(phrase_lower) < 6:
            return False
        
        # Avoid phrases that are mostly stop words
        stop_word_count = sum(1 for word in words if word in self.stop_words)
        if stop_word_count > len(words) / 2:
            return False
        
        # Must contain at least one substantial noun
        substantial_nouns = {
            'tool', 'tools', 'equipment', 'machine', 'system', 'service', 'product',
            'collection', 'kit', 'set', 'supply', 'supplies', 'device', 'instrument',
            'workshop', 'woodworking', 'metalworking', 'crafting', 'manufacturing',
            'router', 'saw', 'drill', 'lathe', 'planer', 'sander', 'dust', 'power'
        }
        
        has_substantial_noun = any(noun in phrase_lower for noun in substantial_nouns)
        
        # Or should be a specific industry term
        industry_terms = any(word in phrase_lower for word in [
            'wood', 'metal', 'craft', 'professional', 'commercial', 'industrial'
        ])
        
        return has_substantial_noun or industry_terms
    
    def _is_high_quality_keyword(self, keyword: str) -> bool:
        """Enhanced quality check for keywords"""
        keyword_lower = keyword.lower().strip()
        words = keyword_lower.split()
        
        # Basic filters
        if (len(keyword_lower) < 3 or 
            keyword_lower.isdigit() or
            not re.match(r'^[a-zA-Z0-9\s\-]+$', keyword)):
            return False
        
        # Single words must be substantial
        if len(words) == 1:
            if (len(keyword_lower) < 5 or 
                keyword_lower in self.stop_words or
                self._is_generic_filler(keyword)):
                return False
        
        # Multi-word phrases should be meaningful
        if len(words) >= 2:
            # Check for too many stop words
            stop_count = sum(1 for word in words if word in self.stop_words)
            if stop_count > len(words) / 2:
                return False
        
        # Must not be a weak fragment
        return not self._is_weak_fragment(keyword)
    
    def _is_weak_fragment(self, keyword: str) -> bool:
        """Check if keyword is a weak grammatical fragment"""
        keyword_lower = keyword.lower().strip()
        
        # Common weak fragments to avoid
        weak_patterns = [
            r'^(find|to|the|and|or|but|with|for|from|through|walk|you)\s',
            r'\s(to|the|and|or|but|with|for|from|through|you)$',
            r'^(create|make|get|take|give|put|set)\s',
            r'(ing|ed|er|est)$',  # Avoid single word endings
        ]
        
        for pattern in weak_patterns:
            if re.search(pattern, keyword_lower):
                return True
        
        # Avoid incomplete phrases
        incomplete_phrases = {
            'find your', 'to create', 'walk you', 'through the', 'get the',
            'make your', 'take the', 'give you', 'put the', 'set up',
            'how to', 'what is', 'why you', 'when you', 'where to'
        }
        
        return keyword_lower in incomplete_phrases
    
    def _apply_final_scoring(self, keywords: Dict[str, float], text: str, url: str) -> Dict[str, float]:
        """Apply final scoring algorithm to keywords"""
        scored_keywords = {}
        text_lower = text.lower()
        text_length = len(text.split())
        
        for keyword, base_score in keywords.items():
            score = base_score
            keyword_lower = keyword.lower()
            
            # Frequency scoring (optimal 1-3% of content)
            frequency = text_lower.count(keyword_lower)
            if frequency > 0:
                frequency_ratio = frequency / text_length * 100
                if 0.5 <= frequency_ratio <= 3.0:
                    score *= 1.3
                elif frequency_ratio > 5.0:
                    score *= 0.6  # Penalize over-optimization
            
            # Word count bonus for specific ranges
            word_count = len(keyword.split())
            if word_count == 2:
                score *= 1.4  # Sweet spot for SEO
            elif word_count == 3:
                score *= 1.2
            elif word_count >= 4:
                score *= 0.9  # Too long
            
            # Commercial intent bonus
            if self._has_commercial_intent(keyword):
                score *= 1.5
            
            # Technical/specific term bonus
            if self._is_technical_or_specific(keyword):
                score *= 1.3
            
            scored_keywords[keyword] = score
        
        return scored_keywords
    
    def _is_technical_or_specific(self, keyword: str) -> bool:
        """Check if keyword is technical or industry-specific"""
        keyword_lower = keyword.lower()
        
        # Technical tool terms
        technical_terms = {
            'router', 'lathe', 'planer', 'jointer', 'bandsaw', 'tablesaw',
            'dust collection', 'woodturning', 'dovetail', 'mortise', 'tenon',
            'carbide', 'diamond', 'spiral', 'flush trim', 'bearing guided'
        }
        
        # Check for technical patterns
        technical_patterns = [
            r'\w+saw\b', r'\w+drill\b', r'\w+router\b', r'\w+lathe\b',
            r'\w+turning\b', r'\w+working\b', r'\w+craft\b'
        ]
        
        # Direct technical term match
        if any(term in keyword_lower for term in technical_terms):
            return True
        
        # Pattern match
        for pattern in technical_patterns:
            if re.search(pattern, keyword_lower):
                return True
        
        return False
    
    def _enhanced_fallback_extraction(self, text: str, max_keywords: int) -> List[str]:
        """Enhanced fallback with better phrase extraction"""
        try:
            # Remove HTML and normalize
            clean_text = re.sub(r'<[^>]+>', ' ', text)
            clean_text = re.sub(r'[^\w\s\-]', ' ', clean_text)
            
            # Extract phrases and single words
            phrases = []
            
            # Split into sentences and extract phrases
            sentences = re.split(r'[.!?]+', clean_text)
            for sentence in sentences:
                words = sentence.lower().split()
                
                # Extract 2-3 word phrases
                for i in range(len(words) - 1):
                    phrase = f"{words[i]} {words[i+1]}"
                    if (len(phrase) > 6 and 
                        not any(stop in phrase for stop in ['the ', 'and ', 'or ']) and
                        not self._is_weak_fragment(phrase)):
                        phrases.append(phrase)
                    
                    # Try 3-word phrases
                    if i < len(words) - 2:
                        phrase3 = f"{words[i]} {words[i+1]} {words[i+2]}"
                        if (len(phrase3) > 10 and
                            not self._is_weak_fragment(phrase3)):
                            phrases.append(phrase3)
            
            # Score and filter phrases
            phrase_scores = {}
            for phrase in phrases:
                if self._is_high_quality_keyword(phrase):
                    phrase_scores[phrase] = clean_text.lower().count(phrase.lower())
            
            # Return top phrases
            return [phrase for phrase, score in 
                   sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)[:max_keywords]]
        
        except Exception as e:
            logging.error(f"Enhanced fallback failed: {e}")
            return []
    
    def _fallback_extraction(self, text: str, max_keywords: int) -> List[str]:
        """Enhanced fallback keyword extraction method"""
        try:
            # Simple frequency-based extraction with better filtering
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            word_freq = Counter(words)
            
            # Filter out stop words and generic terms
            filtered_words = {}
            for word, freq in word_freq.items():
                if (word not in self.stop_words and 
                    len(word) >= 3 and 
                    not self._is_generic_filler(word) and
                    freq >= 2):  # Must appear at least twice
                    filtered_words[word] = freq
            
            # Extract meaningful phrases
            phrases = self._extract_simple_phrases(text)
            for phrase in phrases:
                if not self._is_generic_filler(phrase):
                    filtered_words[phrase] = filtered_words.get(phrase, 0) + 3
            
            # Return top keywords
            return [word for word, freq in 
                   sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:max_keywords]]
        
        except Exception as e:
            logging.error(f"Fallback extraction failed: {str(e)}")
            return []
    
    def _extract_simple_phrases(self, text: str) -> List[str]:
        """Extract simple meaningful phrases for fallback"""
        phrases = []
        sentences = text.split('.')
        
        for sentence in sentences:
            words = sentence.lower().split()
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                if (len(phrase) > 6 and 
                    not any(stop in phrase for stop in ['the ', 'and ', 'or ', 'but ']) and
                    not self._is_generic_filler(phrase)):
                    phrases.append(phrase)
        
        return phrases

# Test function
def test_ai_extraction():
    """Test the AI keyword extractor with woodcraft example"""
    extractor = AIKeywordExtractor()
    
    test_text = """
    <h1>Woodcraft - Quality Woodworking Tools & Supplies</h1>
    <h2>Router Jigs & Templates</h2>
    Welcome to Woodcraft, your premier destination for woodworking tools and supplies. 
    We offer professional router jigs, dovetail saws, dust collection systems, and 
    woodturning lathes. Our woodcraft store features power carving tools, hand tools, 
    and workshop equipment for serious woodworkers. Find router bits, saw blades, 
    and premium lumber for your next project.
    <strong>Professional Dust Collection Systems</strong>
    <b>Woodturning Lathe Collection</b>
    """
    
    keywords = extractor.extract_keywords(test_text, "https://woodcraft.com", 10)
    print(f"Enhanced AI-extracted keywords: {keywords}")
    return keywords

if __name__ == "__main__":
    test_ai_extraction()
