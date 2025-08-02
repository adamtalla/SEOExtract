
import json
import logging
import re
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import math

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
            
            # Train on each example
            for example in training_data:
                keyword = example.get('keyword', '').strip().lower()
                label = example.get('label', '').strip().lower()
                
                if keyword and label in ['good', 'bad']:
                    self._train_on_example(keyword, label)
            
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
        
        # Score based on individual words
        for word in words:
            if len(word) >= 3:
                good_prob = self._calculate_word_probability(word, 'good')
                bad_prob = self._calculate_word_probability(word, 'bad')
                good_score += good_prob
                bad_score += bad_prob
        
        # Score based on patterns
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            good_prob = self._calculate_pattern_probability(bigram, 'good')
            bad_prob = self._calculate_pattern_probability(bigram, 'bad')
            good_score += good_prob * 2  # Weight patterns more heavily
            bad_score += bad_prob * 2
            
            if i < len(words) - 2:
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                good_prob = self._calculate_pattern_probability(trigram, 'good')
                bad_prob = self._calculate_pattern_probability(trigram, 'bad')
                good_score += good_prob * 3  # Weight longer patterns even more
                bad_score += bad_prob * 3
        
        # Score based on structural features
        structural_good, structural_bad = self._calculate_structural_scores(keyword_lower)
        good_score += structural_good
        bad_score += structural_bad
        
        # Convert to probability (0-1 range)
        total_score = good_score + bad_score
        if total_score == 0:
            return 0.5  # Neutral if no patterns match
        
        final_score = good_score / total_score
        return max(0.0, min(1.0, final_score))  # Clamp to 0-1 range
    
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
    
    def classify_keyword(self, keyword: str) -> str:
        """Classify a keyword as 'good' or 'bad'"""
        score = self.calculate_keyword_score(keyword)
        return 'good' if score > 0.5 else 'bad'
    
    def filter_keywords(self, keywords: List[str], threshold: float = 0.6) -> List[str]:
        """Filter keywords, keeping only those above the quality threshold"""
        if not self.is_trained:
            return keywords  # Return all if not trained
        
        filtered = []
        for keyword in keywords:
            score = self.calculate_keyword_score(keyword)
            if score >= threshold:
                filtered.append(keyword)
        
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
            'top_good_words': dict(Counter(self.good_words).most_common(10)),
            'top_bad_words': dict(Counter(self.bad_words).most_common(10)),
            'top_good_patterns': dict(Counter(self.good_patterns).most_common(5)),
            'top_bad_patterns': dict(Counter(self.bad_patterns).most_common(5))
        }

# Global instance
keyword_trainer = SEOKeywordTrainer()

def get_keyword_trainer():
    """Get the global keyword trainer instance"""
    return keyword_trainer
