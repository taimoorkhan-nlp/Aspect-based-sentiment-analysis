__all__ = ['read_data', 'clean_tokenize', 'integer_encode_dataset', 'write_encoded_dataset', 'write_dictionaries']

import string
import json

en_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def clean_tokenize(text):
    clean_text = text.lower()
    # cleaning documents by removing unwanted characters
    clean_text = "".join([char for char in text if char in string.ascii_lowercase])
    # cleaning documents by stopwords
    clean_text = [word for word in text.split(" ") if word not in en_stopwords and len(word) > 2]
    return clean_text

def read_data(file_path):
    rawdata = open(file_path).read()
    #rawdata = df["Title"].tolist()
    return rawdata.split("\n")
    
def integer_encode_dataset(tokenized_documents):
    # Create a dictionary of unique tokens and assign integers
    dictionary = {}
    revdictionary = {}
    index = 0

    #tokenized_documents = [[word for word in doc if word not in esw] for doc in tokenized_documents]

    for doc in tokenized_documents:
        for word in doc:
            if word not in dictionary.keys():
                dictionary[word] = index
                revdictionary[index] = word
                index += 1

    # Replace words in sentences with their corresponding integers
    encoded_documents = [[dictionary[word] for word in doc] for doc in tokenized_documents]
    for doc in encoded_documents:
        for w in doc:
            if w > 3995:
                print(w)
    return encoded_documents, dictionary, revdictionary

def write_encoded_dataset(encoded_documents, file_path):
    toStr = ''
    for endoc in encoded_documents:
        toStr = toStr + '\t'.join(str(item) for item in endoc)
        toStr = toStr + '\n'
    toStr = toStr[:-2]
    file = open(file_path, 'w')
    file.write(toStr)

def write_dictionaries(dictionary, revdictionary, dictionary_path, revdictionary_path):

    with open(dictionary_path, 'w') as file:
        file.write(json.dumps(dictionary))

    with open(revdictionary_path, 'w')as file:
        file.write(json.dumps(revdictionary))
