import requests
import re
import numpy as np
import pandas as pd
import math
from collections import Counter


#input url's for books on PG as .txt files
meditations_url = "https://www.gutenberg.org/cache/epub/2680/pg2680.txt"
walden_url = "https://www.gutenberg.org/cache/epub/205/pg205.txt"

#dowloading the text and cleaning the text with regex
def download_and_clean_text(url):
    response = requests.get(url)
    text = response.text
    #remove PG headers and footers
    #hardcoded the phrases
    start_idx = text.find("START OF THIS PROJECT GUTENBERG EBOOK")
    end_idx = text.find("END OF THIS PROJECT GUTENBERG EBOOK")
    if start_idx != -1 and end_idx != -1:
        #parse through all of the text
        text = text[start_idx:end_idx]
    #using regex to remove the headers, footers, newlines, etc
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

#download and clean using functions
meditations_text = download_and_clean_text(meditations_url)
walden_text = download_and_clean_text(walden_url)

#tokenization
documents = {
    'Meditations': meditations_text.split(),
    'Walden': walden_text.split()
}

#stopwords since they have low tfidf scores anyways
stopwords = set([
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you",
    "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", 
    "all", "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", 
    "when", "make", "can", "like", "no", "just", "him", "know", "take", "person", "into", "year", "your", 
    "good", "some", "could", "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", 
    "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even", "new", 
    "want", "because", "any", "these", "give", "day", "most", "us", "is", "was", "are", "has", "had"
])

#removing stopwords functions
def remove_stopwords(tokenized_text):
    return [word for word in tokenized_text if word not in stopwords]

#applying the stopwords
documents['Meditations'] = remove_stopwords(documents['Meditations'])
documents['Walden'] = remove_stopwords(documents['Walden'])

#create a pandas dataframe with word frequencies
def create_word_freq(doc):
    word_counts = Counter(doc)
    return pd.DataFrame(word_counts.items(), columns=['Word', 'Count'])

#calculate idf
def calculate_idf_single(doc, total_documents=2):
    total_words = len(set(doc))
    word_freq = Counter(doc)
    
    #each word
    idf = {}
    for word, count in word_freq.items():
        #if word in both
        idf[word] = math.log(total_documents / (1 + (1 if word in documents['Meditations'] and word in documents['Walden'] else 0)))
    return idf

#top idf scores
def get_top_idf_words(document_name, document_text, top_n=10):
    idf_scores = calculate_idf_single(document_text)
    word_freq = Counter(document_text)
    
    #sort
    sorted_idf = sorted(idf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    #create df
    return pd.DataFrame({
        'Word': [word for word, idf_value in sorted_idf],
        'Count': [word_freq[word] for word, idf_value in sorted_idf],
        'IDF': [idf_value for word, idf_value in sorted_idf]
    })

top_meditations_idf = get_top_idf_words('Meditations', documents['Meditations'])
top_walden_idf = get_top_idf_words('Walden', documents['Walden'])

print("\nTop IDF Words for Meditations:")
print(top_meditations_idf)

print("\nTop IDF Words for Walden:")
print(top_walden_idf)
