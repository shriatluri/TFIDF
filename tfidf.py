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

# Manually define stopwords or use a predefined list
stopwords = set(["the", "and", "to", "of", "a", "in", "that", "it", "is", "with", "as", "for", "on", "this", "at", "by"])

# Function to remove stopwords
def remove_stopwords(tokenized_text):
    return [word for word in tokenized_text if word not in stopwords]

# Applying stopword removal to the documents
documents['Meditations'] = remove_stopwords(documents['Meditations'])
documents['Walden'] = remove_stopwords(documents['Walden'])

# Function to calculate TF (Term Frequency) for one document
def calculate_tf(doc):
    word_freq = Counter(doc)
    total_words = len(doc)
    
    # Calculate TF for each word
    tf = {word: count / total_words for word, count in word_freq.items()}
    return tf

# Function to calculate IDF (Inverse Document Frequency) specific to a single document
def calculate_idf(doc, total_documents=1):
    word_freq = Counter(doc)
    total_words = len(set(doc))
    
    # Calculate IDF for each word (assuming it appears in only this document)
    idf = {word: math.log(total_documents / (1 + 1)) for word in word_freq}
    return idf

# Function to calculate TF-IDF for one document
def calculate_tfidf(tf, idf):
    tfidf = {word: tf[word] * idf[word] for word in tf}
    return tfidf

# Get the top TF-IDF scores for a document
def get_top_tfidf_words(document_name, document_text, top_n=10):
    # Step 1: Calculate TF (Term Frequency)
    tf = calculate_tf(document_text)
    
    # Step 2: Calculate IDF specific to this document
    idf = calculate_idf(document_text)
    
    # Step 3: Calculate TF-IDF
    tfidf = calculate_tfidf(tf, idf)
    
    # Step 4: Sort by TF-IDF values in descending order and take the top N words
    sorted_tfidf = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Step 5: Create a DataFrame with the top words, their TF-IDF scores, and their TF and IDF
    return pd.DataFrame({
        'Word': [word for word, score in sorted_tfidf],
        'TF-IDF': [score for word, score in sorted_tfidf],
        'TF': [tf[word] for word, score in sorted_tfidf],
        'IDF': [idf[word] for word, score in sorted_tfidf]
    })

# Step 1: Get top TF-IDF words for each document
top_meditations_tfidf = get_top_tfidf_words('Meditations', documents['Meditations'])
top_walden_tfidf = get_top_tfidf_words('Walden', documents['Walden'])

# Step 2: Display the top TF-IDF scores for each document
print("\nTop TF-IDF Words for Meditations:")
print(top_meditations_tfidf)

print("\nTop TF-IDF Words for Walden:")
print(top_walden_tfidf)