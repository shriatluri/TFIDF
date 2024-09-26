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
def create_word_freq_df(documents):
    all_words = list(set(word for doc in documents.values() for word in doc))
    word_freq = pd.DataFrame(0, index=all_words, columns=documents.keys())
    
    for doc_name, doc_words in documents.items():
        word_counts = Counter(doc_words)
        for word, count in word_counts.items():
            word_freq.loc[word, doc_name] = count
            
    return word_freq

# Calculate Term Frequency (TF)
def calculate_tf(word_freq_df):
    tf = word_freq_df.copy()
    for column in tf.columns:
        total_words = tf[column].sum()
        tf[column] = tf[column] / total_words
    return tf

# Calculate Inverse Document Frequency (IDF)
def calculate_idf(word_freq_df):
    N = len(word_freq_df.columns)  # Number of documents
    idf = word_freq_df.copy()
    idf['idf'] = idf.apply(lambda row: math.log(N / (1 + np.count_nonzero(row))), axis=1)
    return idf['idf']

# Calculate TF-IDF
def calculate_tfidf(tf, idf):
    tfidf = tf.copy()
    for column in tf.columns:
        tfidf[column] = tf[column] * idf
    return tfidf

# Output highest TF-IDF scores for each document with details
def get_highest_tfidf_details(tf, idf, tfidf, word_freq_df, top_n=10):
    detailed_results = {}
    for column in tf.columns:
        # Get the top N words with highest TF-IDF scores
        top_words = tfidf[column].sort_values(ascending=False).head(top_n).index
        # Prepare a DataFrame with all details for those words
        details = pd.DataFrame({
            'Count': word_freq_df[column].loc[top_words],
            'TF': tf[column].loc[top_words],
            'IDF': idf.loc[top_words],
            'TF-IDF': tfidf[column].loc[top_words]
        })
        detailed_results[column] = details
    return detailed_results

# Step 1: Create word frequency dataframe
word_freq_df = create_word_freq_df(documents)

# Step 2: Calculate Term Frequency (TF)
tf = calculate_tf(word_freq_df)

# Step 3: Calculate Inverse Document Frequency (IDF)
idf = calculate_idf(word_freq_df)

# Step 4: Calculate TF-IDF
tfidf = calculate_tfidf(tf, idf)

# Step 5: Get the top 10 words with detailed TF-IDF information for each document
detailed_tfidf = get_highest_tfidf_details(tf, idf, tfidf, word_freq_df, top_n=10)

# Display the detailed TF-IDF DataFrame for each document
for doc_name, details_df in detailed_tfidf.items():
    print(f"\nTop TF-IDF Words for {doc_name}:")
    print(details_df)