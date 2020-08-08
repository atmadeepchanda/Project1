import math

import networkx
import numpy

from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


def getrank(document):
    
    sentences = PunktSentenceTokenizer().tokenize(document)

    bow_matrix = CountVectorizer().fit_transform(sentences)
    normalized = TfidfTransformer().fit_transform(bow_matrix)

    similarity_graph = normalized * normalized.T

    nx_graph = networkx.from_scipy_sparse_matrix(similarity_graph)
    values = networkx.pagerank(nx_graph)
    sentence_array = sorted(((values[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    sentence_array = numpy.asarray(sentence_array)
    
    freq_max = float(sentence_array[0][0])
    freq_min = float(sentence_array[len(sentence_array) - 1][0])
    
    temp_array = []
    for i in range(0, len(sentence_array)):
        if freq_max - freq_min == 0:
        	temp_array.append(0)
        else:
        	temp_array.append((float(sentence_array[i][0]) - freq_min) / (freq_max - freq_min))


    threshold = (sum(temp_array) / len(temp_array)) + 0.25
    
    sentence_list = []

    for i in range(0, len(temp_array)):
        if temp_array[i] > threshold:
            sentence_list.append(sentence_array[i][1])

    seq_list = []
    for sentence in sentences:
    	if sentence in sentence_list:
    		seq_list.append(sentence)
    
    return seq_list
