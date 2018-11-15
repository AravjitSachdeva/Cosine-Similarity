# Programming Assignment 1
# Aravjit Sachdeva
# UTA ID - 1001383194

import os
import sys
from nltk.tokenize import RegexpTokenizer
import nltk
nltk.download()
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sys
import math

# Global Variables
token_docs_with_stem = []

stemmer = PorterStemmer()
filename = './debate.txt'
file = open(filename, "r", encoding='UTF-8')
doc = file.read()
file.close()

# Split corpus into documents
list_of_docs = doc.split('\n')
for i in list_of_docs:
    if len(i) == 0:
        list_of_docs.remove(i)

def tokenizer_function(string_to_tokenize):    #Tokenizes given string
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')               
    tokenized_string = tokenizer.tokenize(string_to_tokenize)
    return tokenized_string

def remove_stopwords(tokens_with_stopwords):   # remove stopwords from list of tokens
    stopwords_list = stopwords.words('english')
    tokens_without_stopwords = []
    for j in tokens_with_stopwords:
        if j not in stopwords_list:
            tokens_without_stopwords.append(j)
    return tokens_without_stopwords

def stemming_function(token_without_stopwords): # performs stemming on each token
    stemmer = PorterStemmer()
    stemmed_doc = []
    for i in token_without_stopwords:
        stemmed_doc.append(stemmer.stem(i))

        
    return stemmed_doc
def get_df(input_token): # this function determines the document frequency i.e number of documents in which given token occurs
    df = 0
    for i in range(len(token_docs_with_stem)):
        for j in set(token_docs_with_stem[i]):
            if j == input_token:
                df = df+1
                break
    return df
def getidf(input_token):    # this function determines the idf value for a given token
    df = 0 # df is the number of documents in which given token is present
    
    #print(list_of_docs[0])
    
    df = get_df(input_token)
    # Calculate inverse document frequency
    if df==0:
        idf = -1
    else:
        idf = math.log10(len(token_docs_with_stem)/df)
        
    return idf
    
def getqvec(qstring):  # this function returns a query vector with tf-idf weights for each token 
    #Tokenize 
    qstring = qstring.lower()
    tokenized_string  = tokenizer_function(qstring)
    #remove stop words
    string_without_stopwords = remove_stopwords(tokenized_string)
    #Stemming
    stemmed_string = stemming_function(string_without_stopwords)
    token_tfidf_map = {}
    # Determining tf weights
    for i in stemmed_string:
        if i in token_tfidf_map:
            token_tfidf_map[i] = token_tfidf_map[i] + 1

        else:
            token_tfidf_map[i] = 1
    
    for i in token_tfidf_map:
        token_tfidf_map[i] = 1 + math.log10(token_tfidf_map[i])
    # Determining tf-idf weights
    for i in token_tfidf_map:
        if get_df(i) != 0:  
            token_tfidf_map[i] = token_tfidf_map[i] * getidf(i)
        else:
            token_tfidf_map[i] = token_tfidf_map[i] * math.log10(len(token_docs_with_stem))
    # Normalize query vector
    denominator = 0
    for i in token_tfidf_map:
        denominator = denominator + (token_tfidf_map[i]**2)
    
    
    denominator = math.sqrt(denominator)
    
    for i in token_tfidf_map:
        token_tfidf_map[i] = token_tfidf_map[i]/denominator
    
    return token_tfidf_map
def query(string):          
# this function returns the paragraph and cosine similarity in the transcript that has the highest cosine similarity score with
#respect to qstring
    string = string.lower()
    query_map = getqvec(string) 

    max_cosine_similiarity =0
    cosine_similarity = 0
    index = 0
    

    global list_of_docs
    i = 0
    count = 0
    # Split corpus into documents
    for i in range(len(list_of_docs)):
        qvec_paragraph = getqvec(list_of_docs[i])
        cosine_similarity = 0
        for j in query_map:
            if j in qvec_paragraph:
                # finding cosine similarity for each document
                cosine_similarity = cosine_similarity + (query_map[j]*qvec_paragraph[j]) 
                
                
        if cosine_similarity > max_cosine_similiarity:
            max_cosine_similiarity = cosine_similarity # finding maximum value of cosine similarity
            index = i
    if max_cosine_similiarity == 0:
        return 'No match' + '\n' + str(max_cosine_similiarity)
    # returning the string  with maximum cosine similarity and its score
    return_string = list_of_docs[index] + '\n' + str(max_cosine_similiarity) 
    return return_string
def main():
    
    #Extracting given file

    
    global list_of_docs
    #Tokenize each document
    tokenized_list_of_docs = []

    for i in list_of_docs:
        tokenized_list_of_docs.append(tokenizer_function(i))
    
    
    #Remove stop words from each document
    token_docs_without_stopwords = []

    for i in range(len(tokenized_list_of_docs)):
        token_docs_without_stopwords.append(remove_stopwords(tokenized_list_of_docs[i]))
   
    # perform stemming on each token in the document
    for i in token_docs_without_stopwords:
        token_docs_with_stem.append(stemming_function(i))
        
    #Test here
    
    
    #print("%.4f" % getidf(stemmer.stem("hispanic")))
    #print(getqvec('clinton first amendment kavanagh'))
    #print(query('The alternative, as cruz has proposed, is to deport 11 million people from this country'))
    
main()