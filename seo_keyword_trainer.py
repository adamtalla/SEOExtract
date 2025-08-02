
import json
import logging
import re
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import math
from difflib import SequenceMatcher

class SEOKeywordTrainer:
    """Lightweight AI system to learn and classify SEO keyword patterns"""
    
    def __init__(self, training_file: str = "seo_training_data.json"):
        self.training_file = training_file
        self.good_patterns = defaultdict(int)
        self.bad_patterns = defaultdict(int)
        self.good_words = defaultdict(int)
        self.bad_words = defaultdict(int)
        self.good_structures = defaultdict(int)
        self.bad_structures = defaultdict(int)
        self.total_good = 0
        self.total_bad = 0
        self.is_trained = False
        
        # Create lookup dictionaries for exact matching
        self.keyword_labels = {}  # keyword -> label mapping
        self.good_keywords_set = set()
        self.bad_keywords_set = set()
        
        # Load and train on existing data
        self.load_and_train()
    
    def load_and_train(self):
        """Load training data and build pattern recognition models"""
        try:
            with open(self.training_file, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
            
            logging.info(f"Loaded {len(training_data)} training examples")
            
            # Reset counters
            self.good_patterns.clear()
            self.bad_patterns.clear()
            self.good_words.clear()
            self.bad_words.clear()
            self.good_structures.clear()
            self.bad_structures.clear()
            self.total_good = 0
            self.total_bad = 0
            
            # Train on each example and build lookup dictionaries
            for example in training_data:
                keyword = example.get('keyword', '').strip().lower()
                label = example.get('label', '').strip().lower()
                
                if keyword and label in ['good', 'bad']:
                    self._train_on_example(keyword, label)
                    
                    # Build lookup dictionaries for efficient filtering
                    self.keyword_labels[keyword] = label
                    if label == 'good':
                        self.good_keywords_set.add(keyword)
                    else:
                        self.bad_keywords_set.add(keyword)
            
            self.is_trained = True
            logging.info(f"Training complete: {self.total_good} good, {self.total_bad} bad examples")
            
        except FileNotFoundError:
            logging.warning(f"Training file {self.training_file} not found. Using default patterns.")
            self.is_trained = False
        except Exception as e:
            logging.error(f"Error loading training data: {str(e)}")
            self.is_trained = False
    
    def _train_on_example(self, keyword: str, label: str):
        """Train the model on a single keyword example"""
        if label == 'good':
            self.total_good += 1
            word_counter = self.good_words
            pattern_counter = self.good_patterns
            structure_counter = self.good_structures
        else:
            self.total_bad += 1
            word_counter = self.bad_words
            pattern_counter = self.bad_patterns
            structure_counter = self.bad_structures
        
        # Extract individual words
        words = keyword.split()
        for word in words:
            if len(word) >= 3:
                word_counter[word] += 1
        
        # Extract patterns (bigrams, trigrams)
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            pattern_counter[bigram] += 1
            
            if i < len(words) - 2:
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                pattern_counter[trigram] += 1
        
        # Extract structural features
        structure_counter[f"word_count_{len(words)}"] += 1
        structure_counter[f"char_count_{len(keyword)}"] += 1
        
        # Specific patterns
        if any(word in keyword for word in ['service', 'services', 'professional', 'emergency']):
            structure_counter["has_service_terms"] += 1
        
        if any(word in keyword for word in ['tool', 'tools', 'equipment', 'system', 'systems']):
            structure_counter["has_product_terms"] += 1
        
        if re.search(r'\b(click|find|get|make|take|call)\b', keyword):
            structure_counter["has_action_verbs"] += 1
        
        if re.search(r'\b(here|there|now|your|the)\b', keyword):
            structure_counter["has_filler_words"] += 1
    
    def calculate_keyword_score(self, keyword: str) -> float:
        """Calculate a quality score for a keyword based on learned patterns"""
        if not self.is_trained or (self.total_good == 0 and self.total_bad == 0):
            return 0.5  # Neutral score if not trained
        
        keyword_lower = keyword.strip().lower()
        words = keyword_lower.split()
        
        good_score = 0.0
        bad_score = 0.0
        
        # Enhanced word-level scoring with more sophisticated weighting
        for word in words:
            if len(word) >= 3:
                good_prob = self._calculate_word_probability(word, 'good')
                bad_prob = self._calculate_word_probability(word, 'bad')
                
                # Weight words by length and importance
                word_weight = min(2.0, len(word) / 5.0)  # Longer words get more weight
                good_score += good_prob * word_weight
                bad_score += bad_prob * word_weight
        
        # Enhanced pattern scoring with confidence intervals
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            good_prob = self._calculate_pattern_probability(bigram, 'good')
            bad_prob = self._calculate_pattern_probability(bigram, 'bad')
            
            # Apply confidence weighting based on training data size
            confidence = min(1.0, (self.total_good + self.total_bad) / 100.0)
            good_score += good_prob * 2.5 * confidence
            bad_score += bad_prob * 2.5 * confidence
            
            if i < len(words) - 2:
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                good_prob = self._calculate_pattern_probability(trigram, 'good')
                bad_prob = self._calculate_pattern_probability(trigram, 'bad')
                good_score += good_prob * 3.5 * confidence
                bad_score += bad_prob * 3.5 * confidence
        
        # Enhanced structural scoring
        structural_good, structural_bad = self._calculate_structural_scores(keyword_lower)
        good_score += structural_good * 1.5
        bad_score += structural_bad * 1.5
        
        # Semantic pattern scoring for better accuracy
        semantic_good, semantic_bad = self._calculate_semantic_scores(keyword_lower)
        good_score += semantic_good
        bad_score += semantic_bad
        
        # Convert to probability with smoothing
        total_score = good_score + bad_score
        if total_score == 0:
            return 0.5  # Neutral if no patterns match
        
        # Apply sigmoid function for smoother probability distribution
        raw_ratio = good_score / total_score
        final_score = 1 / (1 + math.exp(-10 * (raw_ratio - 0.5)))
        
        return max(0.0, min(1.0, final_score))
    
    def _calculate_word_probability(self, word: str, label: str) -> float:
        """Calculate probability of a word being good/bad"""
        if label == 'good':
            word_count = self.good_words.get(word, 0)
            total_count = self.total_good
        else:
            word_count = self.bad_words.get(word, 0)
            total_count = self.total_bad
        
        if total_count == 0:
            return 0.0
        
        # Laplace smoothing
        return (word_count + 1) / (total_count + len(self.good_words) + len(self.bad_words))
    
    def _calculate_pattern_probability(self, pattern: str, label: str) -> float:
        """Calculate probability of a pattern being good/bad"""
        if label == 'good':
            pattern_count = self.good_patterns.get(pattern, 0)
            total_count = self.total_good
        else:
            pattern_count = self.bad_patterns.get(pattern, 0)
            total_count = self.total_bad
        
        if total_count == 0:
            return 0.0
        
        # Laplace smoothing
        return (pattern_count + 1) / (total_count + len(self.good_patterns) + len(self.bad_patterns))
    
    def _calculate_structural_scores(self, keyword: str) -> Tuple[float, float]:
        """Calculate structural feature scores"""
        words = keyword.split()
        good_structural = 0.0
        bad_structural = 0.0
        
        # Word count patterns
        word_count_key = f"word_count_{len(words)}"
        good_structural += self.good_structures.get(word_count_key, 0) / max(self.total_good, 1)
        bad_structural += self.bad_structures.get(word_count_key, 0) / max(self.total_bad, 1)
        
        # Character count patterns
        char_count_key = f"char_count_{len(keyword)}"
        good_structural += self.good_structures.get(char_count_key, 0) / max(self.total_good, 1)
        bad_structural += self.bad_structures.get(char_count_key, 0) / max(self.total_bad, 1)
        
        # Semantic patterns
        if any(word in keyword for word in ['service', 'services', 'professional', 'emergency']):
            good_structural += self.good_structures.get("has_service_terms", 0) / max(self.total_good, 1)
            bad_structural += self.bad_structures.get("has_service_terms", 0) / max(self.total_bad, 1)
        
        if any(word in keyword for word in ['tool', 'tools', 'equipment', 'system', 'systems']):
            good_structural += self.good_structures.get("has_product_terms", 0) / max(self.total_good, 1)
            bad_structural += self.bad_structures.get("has_product_terms", 0) / max(self.total_bad, 1)
        
        if re.search(r'\b(click|find|get|make|take|call)\b', keyword):
            good_structural += self.good_structures.get("has_action_verbs", 0) / max(self.total_good, 1)
            bad_structural += self.bad_structures.get("has_action_verbs", 0) / max(self.total_bad, 1)
        
        if re.search(r'\b(here|there|now|your|the)\b', keyword):
            good_structural += self.good_structures.get("has_filler_words", 0) / max(self.total_good, 1)
            bad_structural += self.bad_structures.get("has_filler_words", 0) / max(self.total_bad, 1)
        
        return good_structural, bad_structural
    
    def _calculate_semantic_scores(self, keyword: str) -> Tuple[float, float]:
        """Calculate semantic scores based on advanced pattern recognition"""
        good_semantic = 0.0
        bad_semantic = 0.0
        
        # Advanced semantic patterns for good keywords
        good_semantic_patterns = [
            # Service-oriented patterns
            (r'\b\w+\s+(?:services?|solutions?|repair|installation|maintenance)\b', 2.0),
            (r'\b(?:professional|certified|licensed|experienced)\s+\w+', 1.8),
            (r'\b(?:emergency|24\s*hour|same\s*day|fast|quick)\s+\w+', 1.9),
            
            # Product-oriented patterns
            (r'\b(?:best|top|quality|premium)\s+\w+(?:\s+\w+)*\b', 1.5),
            (r'\b\w+\s+(?:tools?|equipment|systems?|supplies?)\b', 1.7),
            
            # Location-based patterns
            (r'\b\w+\s+(?:near\s+me|in\s+\w+|area|local)\b', 1.6),
            
            # Educational/informational patterns
            (r'\bhow\s+to\s+\w+(?:\s+\w+)*\b', 1.4),
            (r'\b(?:guide|tutorial|tips|course)\s+(?:for|to)\s+\w+', 1.3),
            
            # Commercial intent patterns
            (r'\b(?:buy|purchase|order|hire|book)\s+\w+', 1.8),
            (r'\b(?:affordable|cheap|discount|deal)\s+\w+', 1.5),
        ]
        
        # Advanced semantic patterns for bad keywords
        bad_semantic_patterns = [
            # Generic action patterns
            (r'\b(?:click|find|get|make|take)\s+(?:here|now|today|your|the)\b', 2.5),
            (r'\bwalk\s+you\s+through\b', 3.0),
            (r'\b(?:to|the|and|or|but)\s+\w+\s*$', 2.0),
            
            # Spam/promotional patterns
            (r'\b(?:guaranteed|instant|miracle|secret|breakthrough)\b', 3.0),
            (r'\b(?:amazing|incredible|shocking|unbelievable)\s+\w+', 2.8),
            (r'\b(?:act\s+now|limited\s+time|special\s+offer|exclusive)\b', 2.9),
            
            # Weak fragments
            (r'^\w+\s+(?:ing|ed|er|est)$', 2.2),
            (r'\b(?:buy\s+now|click\s+here|sign\s+up|free\s+trial)\b', 2.7),
            
            # Overly generic terms
            (r'\b(?:best\s+price|lowest\s+rate|cheap\s+and|easy\s+money)\b', 2.5),
        ]
        
        # Score against good patterns
        for pattern, weight in good_semantic_patterns:
            if re.search(pattern, keyword, re.IGNORECASE):
                good_semantic += weight * (self.total_good / max(self.total_good + self.total_bad, 1))
        
        # Score against bad patterns
        for pattern, weight in bad_semantic_patterns:
            if re.search(pattern, keyword, re.IGNORECASE):
                bad_semantic += weight * (self.total_bad / max(self.total_good + self.total_bad, 1))
        
        return good_semantic, bad_semantic
    
    def classify_keyword(self, keyword: str) -> str:
        """Classify a keyword as 'good' or 'bad'"""
        score = self.calculate_keyword_score(keyword)
        return 'good' if score > 0.5 else 'bad'
    
    def filter_keywords(self, keywords: List[str], threshold: float = 0.55) -> List[str]:
        """Filter keywords, keeping only those above the quality threshold"""
        if not self.is_trained:
            return keywords  # Return all if not trained
        
        filtered = []
        scored_keywords = []
        
        # Calculate scores for all keywords
        for keyword in keywords:
            score = self.calculate_keyword_score(keyword)
            scored_keywords.append((keyword, score))
        
        # Sort by score to get distribution
        scored_keywords.sort(key=lambda x: x[1], reverse=True)
        
        # Adaptive threshold based on score distribution
        if len(scored_keywords) > 10:
            # Use a more permissive threshold for larger datasets
            scores = [score for _, score in scored_keywords]
            median_score = scores[len(scores) // 2]
            adaptive_threshold = max(threshold, median_score * 0.9)
        else:
            adaptive_threshold = threshold
        
        # Filter using adaptive threshold
        for keyword, score in scored_keywords:
            if score >= adaptive_threshold:
                filtered.append(keyword)
        
        # Ensure we return at least some keywords if input was substantial
        if len(filtered) < max(3, len(keywords) * 0.3) and keywords:
            # Return top 50% if strict filtering removed too many
            half_point = len(scored_keywords) // 2
            filtered = [kw for kw, score in scored_keywords[:half_point]]
        
        return filtered
    
    def rank_keywords(self, keywords: List[str]) -> List[Tuple[str, float]]:
        """Rank keywords by their quality scores"""
        scored_keywords = []
        for keyword in keywords:
            score = self.calculate_keyword_score(keyword)
            scored_keywords.append((keyword, score))
        
        # Sort by score (descending)
        scored_keywords.sort(key=lambda x: x[1], reverse=True)
        return scored_keywords
    
    def add_training_example(self, keyword: str, label: str):
        """Add a new training example and retrain"""
        try:
            # Load existing data
            try:
                with open(self.training_file, 'r', encoding='utf-8') as f:
                    training_data = json.load(f)
            except FileNotFoundError:
                training_data = []
            
            # Add new example
            new_example = {"keyword": keyword.strip(), "label": label.strip().lower()}
            training_data.append(new_example)
            
            # Save updated data
            with open(self.training_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
            
            # Retrain
            self.load_and_train()
            logging.info(f"Added training example: {keyword} -> {label}")
            return True
            
        except Exception as e:
            logging.error(f"Error adding training example: {str(e)}")
            return False
    
    def get_training_stats(self) -> Dict:
        """Get statistics about the training data"""
        return {
            'total_examples': self.total_good + self.total_bad,
            'good_examples': self.total_good,
            'bad_examples': self.total_bad,
            'is_trained': self.is_trained,
            'unique_good_words': len(self.good_words),
            'unique_bad_words': len(self.bad_words),
            'unique_good_patterns': len(self.good_patterns),
            'unique_bad_patterns': len(self.bad_patterns),
            'top_good_words': dict(Counter(self.good_words).most_common(10)),
            'top_bad_words': dict(Counter(self.bad_words).most_common(10)),
            'top_good_patterns': dict(Counter(self.good_patterns).most_common(5)),
            'top_bad_patterns': dict(Counter(self.bad_patterns).most_common(5))
        }
    
    def get_fuzzy_match_score(self, keyword1: str, keyword2: str) -> float:
        """Calculate similarity score between two keywords using fuzzy matching"""
        return SequenceMatcher(None, keyword1.lower(), keyword2.lower()).ratio()
    
    def find_similar_keyword(self, keyword: str, threshold: float = 0.85) -> Tuple[str, str]:
        """Find similar keyword in training data using fuzzy matching"""
        keyword_lower = keyword.lower()
        best_match = None
        best_score = 0.0
        best_label = None
        
        # Check against all training keywords
        for training_keyword, label in self.keyword_labels.items():
            score = self.get_fuzzy_match_score(keyword_lower, training_keyword)
            if score >= threshold and score > best_score:
                best_score = score
                best_match = training_keyword
                best_label = label
        
        return best_match, best_label
    
    def get_keyword_label_direct(self, keyword: str, use_fuzzy: bool = True) -> str:
        """Get direct label for keyword with optional fuzzy matching"""
        keyword_lower = keyword.lower().strip()
        
        # Try exact match first
        if keyword_lower in self.keyword_labels:
            return self.keyword_labels[keyword_lower]
        
        # Try fuzzy matching if enabled
        if use_fuzzy:
            similar_keyword, label = self.find_similar_keyword(keyword_lower, threshold=0.85)
            if similar_keyword and label:
                return label
        
        return 'unknown'
    
    def score_keyword_direct(self, keyword: str, use_fuzzy: bool = True) -> int:
        """Score keyword directly: +1 for good, -1 for bad, 0 for unknown"""
        label = self.get_keyword_label_direct(keyword, use_fuzzy)
        
        if label == 'good':
            return 1
        elif label == 'bad':
            return -1
        else:
            return 0
    
    def filter_keywords_direct(self, keywords: List[str], remove_bad: bool = True, use_fuzzy: bool = True) -> List[str]:
        """Filter keywords by removing bad ones and keeping good/unknown ones"""
        if not self.is_trained:
            return keywords
        
        filtered_keywords = []
        
        for keyword in keywords:
            label = self.get_keyword_label_direct(keyword, use_fuzzy)
            
            if remove_bad and label == 'bad':
                continue  # Skip bad keywords
            
            filtered_keywords.append(keyword)
        
        return filtered_keywords
    
    def score_and_sort_keywords(self, keywords: List[str], use_fuzzy: bool = True) -> List[Tuple[str, int]]:
        """Score keywords and return them sorted by score (highest first)"""
        scored_keywords = []
        
        for keyword in keywords:
            score = self.score_keyword_direct(keyword, use_fuzzy)
            # Also factor in the AI-based probability score
            ai_score = self.calculate_keyword_score(keyword)
            # Combine direct score with AI score (weighted)
            combined_score = score * 0.7 + (ai_score - 0.5) * 2 * 0.3  # Normalize AI score to -1 to +1 range
            scored_keywords.append((keyword, combined_score))
        
        # Sort by score (highest first)
        scored_keywords.sort(key=lambda x: x[1], reverse=True)
        return scored_keywords
    
    def reload_training_data(self):
        """Reload training data from file (useful after adding new examples)"""
        logging.info("Reloading training data...")
        self.load_and_train()
        stats = self.get_training_stats()
        logging.info(f"Training data reloaded: {stats['total_examples']} total examples")
        logging.info(f"Good examples: {stats['good_examples']}, Bad examples: {stats['bad_examples']}")
        logging.info(f"Unique patterns: {stats['unique_good_patterns']} good, {stats['unique_bad_patterns']} bad")
        return stats
    
    def get_filtering_stats(self, keywords: List[str]) -> Dict:
        """Get detailed statistics about keyword filtering"""
        if not keywords:
            return {'total': 0, 'good': 0, 'bad': 0, 'unknown': 0}
        
        stats = {'total': len(keywords), 'good': 0, 'bad': 0, 'unknown': 0}
        
        for keyword in keywords:
            label = self.get_keyword_label_direct(keyword, use_fuzzy=True)
            if label == 'good':
                stats['good'] += 1
            elif label == 'bad':
                stats['bad'] += 1
            else:
                stats['unknown'] += 1
        
        return stats

# Global instance
keyword_trainer = SEOKeywordTrainer()

def get_keyword_trainer():
    """Get the global keyword trainer instance"""
    return keyword_trainer
