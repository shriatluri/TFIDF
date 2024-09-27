import requests
import re
import numpy as np
import pandas as pd
import math
from collections import Counter
import matplotlib.pyplot as plt

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

#calculating the tf
def calculate_tf(doc):
    word_freq = Counter(doc)
    total_words = len(doc)
    #actual calculation
    tf = {word: count / total_words for word, count in word_freq.items()}
    return tf

#function to calculate idf for the specific document
def calculate_idf(doc, total_documents=1):
    word_freq = Counter(doc)
    total_words = len(set(doc))
    
    #calculate idf for each word
    idf = {word: math.log(total_documents / (1 + 1)) for word in word_freq}
    return idf

#function to calculate tfidf for each word
def calculate_tfidf(tf, idf):
    tfidf = {word: tf[word] * idf[word] for word in tf}
    return tfidf

#function to get the top TFIDF words
def get_top_tfidf_words(docuemnt_name, document_text, top_n=10):
    #calculating tf
    tf = calculate_tf(document_text)
    #calculating count
    word_freq = Counter(document_text)
    #calculating idf
    idf = calculate_idf(document_text)
    #calculating tfidf
    tfidf = calculate_tfidf(tf, idf)
    #sorting the values
    sorted_tfidf = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:top_n]
    #using pandas to create the df of all of the values
    return pd.DataFrame({
        'Word': [word for word, score in sorted_tfidf],
        'Count': [word_freq[word] for word, value in sorted_tfidf],
        'TF-IDF': [score for word, score in sorted_tfidf],
        'TF': [tf[word] for word, score in sorted_tfidf],
        'IDF': [idf[word] for word, score in sorted_tfidf]
    })

#get the top scores using the function
top_meditations_tfidf = get_top_tfidf_words('Meditations', documents['Meditations'])
top_walden_tfidf = get_top_tfidf_words('Walden', documents['Walden'])

#displaying the top words for meditations
print("\nTop TF-IDF Words for Meditations:")
print(top_meditations_tfidf)
#displaying the top words for walden
print("\nTop TF-IDF Words for Walden:")
print(top_walden_tfidf)

#matplotlib part
#function to create a bar plot for the top TF-IDF words
def plot_tfidf_scores(document_name, tfidf_df):
    tfidf_df = tfidf_df.sort_values(by='TF-IDF', ascending=False)
    plt.figure(figsize=(8, 5))
    plt.barh(tfidf_df['Word'], tfidf_df['TF-IDF'], color='skyblue')
    plt.xlabel('TF-IDF Score')
    plt.ylabel('Word')
    plt.title(f'Top TF-IDF Words for {document_name}')
    plt.gca().invert_yaxis()
    plt.show()

plot_tfidf_scores('Meditations', top_meditations_tfidf)
plot_tfidf_scores('Walden', top_walden_tfidf)
