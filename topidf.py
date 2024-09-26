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

# Create a pandas dataframe with word frequencies
def create_word_freq(doc):
    word_counts = Counter(doc)
    return pd.DataFrame(word_counts.items(), columns=['Word', 'Count'])

# Calculate Inverse Document Frequency (IDF) for one document at a time
def calculate_idf_single(doc, total_documents=2):
    total_words = len(set(doc))
    word_freq = Counter(doc)
    
    # Calculate IDF for each word
    idf = {}
    for word, count in word_freq.items():
        # If word appears in both documents, adjust IDF accordingly
        idf[word] = math.log(total_documents / (1 + (1 if word in documents['Meditations'] and word in documents['Walden'] else 0)))
    return idf

# Get the top IDF scores and their counts for each document
def get_top_idf_words(document_name, document_text, top_n=10):
    idf_scores = calculate_idf_single(document_text)
    word_freq = Counter(document_text)
    
    # Sort by IDF values in descending order and take the top N words
    sorted_idf = sorted(idf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Create a DataFrame with the top words, their IDF scores, and their frequencies (counts)
    return pd.DataFrame({
        'Word': [word for word, idf_value in sorted_idf],
        'Count': [word_freq[word] for word, idf_value in sorted_idf],
        'IDF': [idf_value for word, idf_value in sorted_idf]
    })

# Step 1: Get top IDF words with counts for each document
top_meditations_idf = get_top_idf_words('Meditations', documents['Meditations'])
top_walden_idf = get_top_idf_words('Walden', documents['Walden'])

# Step 2: Display the top IDF scores with word counts for each document
print("\nTop IDF Words for Meditations:")
print(top_meditations_idf)

print("\nTop IDF Words for Walden:")
print(top_walden_idf)