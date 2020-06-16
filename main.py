from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation 
from math import log10
import numpy as np

def main():
    sample = input("Sample text: ")
    print('-'*100)
    test = input("Verified text: ")    

    sampleVec = toVector(sample)
    testVec = toVector(test)

    res = calPercent(sampleVec, testVec)
    print('-'*100)
    print('Percent of match is about {:0.2f}%'.format(res))
    print('-'*100)

def calPercent(sample, test):
    sampleSet = list(set(sample))
    testSet = list(set(test))
    wordList = list(set(sample + test))
    size = len(wordList)

    sample_tf = cal_tf(sampleSet, sample, wordList) 
    test_tf = cal_tf(testSet, test, wordList)
    
    idfs = cal_idf([sampleSet, testSet], wordList)

    sample_tf_idf = np.array(sample_tf) * np.array(idfs)
    test_tf_idf = np.array(test_tf) * np.array(idfs)

    cousine_similarity = sum(sample_tf_idf * test_tf_idf) / (np.linalg.norm(sample_tf_idf) * np.linalg.norm(test_tf_idf))

    return cousine_similarity * 100        

def cal_idf(textSets, wordList):
    n = 2
    idf = []

    for w in wordList:
        count = 0
        for s in textSets:
            if w in s:
                count += 1
        idf.append(1 + log10(n/count))

    return idf

def cal_tf(textSet, text, wordList):
    res = [0 for w in wordList]
    for w in textSet:
        count = 0
        for w1 in text:
            if w == w1:
                count += 1
        res[wordList.index(w)] = count / len(textSet)

    return res

def toVector(text):
    stopWords = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filteredTokens = []
    lemmatizer = WordNetLemmatizer()
    
    for token in tokens:
        if token not in stopWords and token not in punctuation:
            lemma = lemmatizer.lemmatize(token)
            filteredTokens.append(lemma)
    
    return filteredTokens


if __name__ == "__main__":
    main()