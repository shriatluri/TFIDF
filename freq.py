import requests
import re
import numpy as np
import pandas as pd
import math
from collections import Counter

# URLs for the books
meditations_url = "https://www.gutenberg.org/cache/epub/2680/pg2680.txt"
walden_url = "https://www.gutenberg.org/cache/epub/205/pg205.txt"

# Function to download and clean text
def download_and_clean_text(url):
    response = requests.get(url)
    text = response.text
    
    # Remove Project Gutenberg headers and footers
    start_idx = text.find("START OF THIS PROJECT GUTENBERG EBOOK")
    end_idx = text.find("END OF THIS PROJECT GUTENBERG EBOOK")
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
    
    # Remove headers, footers, and any non-text data
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with space
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabet characters
    text = text.lower()  # Convert text to lowercase
    return text

# Download and clean texts
meditations_text = download_and_clean_text(meditations_url)
walden_text = download_and_clean_text(walden_url)

# Tokenize the texts
documents = {
    'Meditations': meditations_text.split(),
    'Walden': walden_text.split()
}

# Define the top 100 most common English words as stopwords
top_100_stopwords = set([
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you",
    "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", 
    "all", "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", 
    "when", "make", "can", "like", "no", "just", "him", "know", "take", "person", "into", "year", "your", 
    "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", 
    "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", 
    "want", "because", "any", "these", "give", "day", "most", "us", "is", "was", "are", "has", "had"
])

# Function to remove stopwords
def remove_stopwords(tokenized_text):
    return [word for word in tokenized_text if word not in top_100_stopwords]

# Applying stopword removal to the documents
documents['Meditations'] = remove_stopwords(documents['Meditations'])
documents['Walden'] = remove_stopwords(documents['Walden'])

# Function to calculate TF (Term Frequency) for one document
def calculate_tf(doc):
    word_freq = Counter(doc)
    total_words = len(doc)
    
    # Calculate TF for each word
    tf = {word: count / total_words for word, count in word_freq.items()}
    return tf, word_freq  # Return both TF and word counts

# Get the top TF (highest frequency) words with word counts for a document
def get_top_tf_words(document_name, document_text, top_n=10):
    # Step 1: Calculate TF (Term Frequency) and word count
    tf, word_count = calculate_tf(document_text)
    
    # Step 2: Sort by frequency (TF) in descending order and take the top N words
    sorted_tf = sorted(tf.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Step 3: Create a DataFrame with the top words, their counts, and TF scores
    return pd.DataFrame({
        'Word': [word for word, score in sorted_tf],
        'Count': [word_count[word] for word, score in sorted_tf],
        'TF': [score for word, score in sorted_tf]
    })

# Step 1: Get top TF words for each document
top_meditations_tf = get_top_tf_words('Meditations', documents['Meditations'])
top_walden_tf = get_top_tf_words('Walden', documents['Walden'])

# Step 2: Display the top TF scores with word counts for each document
print("\nTop TF Words for Meditations:")
print(top_meditations_tf)

print("\nTop TF Words for Walden:")
print(top_walden_tf)