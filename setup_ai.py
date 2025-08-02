
#!/usr/bin/env python3
"""
Setup script for AI keyword extraction dependencies
Run this once to download required NLTK data
"""

import nltk
import logging

def setup_nltk_data():
    """Download required NLTK data for AI keyword extraction"""
    print("Setting up AI keyword extraction dependencies...")
    
    # Required NLTK data packages
    packages = [
        'punkt',           # Sentence and word tokenization
        'stopwords',       # Stop words corpus
        'averaged_perceptron_tagger',  # POS tagger
        'maxent_ne_chunker',  # Named entity chunker
        'words',           # Word corpus
        'wordnet'          # WordNet lexical database
    ]
    
    for package in packages:
        try:
            print(f"Downloading {package}...")
            nltk.download(package, quiet=True)
            print(f"✓ {package} downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download {package}: {str(e)}")
    
    print("\nAI keyword extraction setup complete!")
    print("You can now use the enhanced AI-powered keyword extraction.")

if __name__ == "__main__":
    setup_nltk_data()
