# Kathleen Schaefer and Johnathan Ashley

#pip install -U numpy scipy scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer #
import numpy as np
import sentifier as st
import os
import xml.etree.ElementTree as ET
import random
import sys

"""
sent1 = "this is the first sentence of paragraph one ."
sent2 = "this is the last sentence of paragraph one ."
sent3 = "this is a highly important sentence with highly important facts ."
sent4 = "nothing here a and is how the this I me no you it is was ."
sent5 = "distinguishing words corpus panda document cat dog house ."
example = [sent1, sent2, sent3, sent4, sent5]
#example = [[sent1.split(), sent2.split()],[sent3.split()]]

corpSent1 = "Look at this wonderful with corpus of words "
corpSent2 = "Isn ' t it so of wonderful ?"
corpSent3 = "Here is another of of corpus with common words a with of to a."
corpSent4 = "This one has two paragraph with of to the of this a."
corpus = [[[corpSent1.split(), corpSent2.split()]],
	[[corpSent3.split()],[corpSent4.split()]]]
"""

def tfidfRank(d, toScore):
	""" calculates the average tf-idf score of every sentence
	in the example for the given corpus """

	scores = []
	"""
	for paragraph in toScore:
		parScores = []
		for sentence in paragraph:
			sentScore = 0.0
			numHits = 0.0
			for word in sentence:
				if word in d:
					sentScore += d[word]
					numHits += 1.0
			if sentScore > 0:
				sentScore = sentScore/numHits
			parScores.append(sentScore)
		scores.append(parScores)
	"""

	for sentence in toScore:
		sentScore = 0.0
		numHits = 0.0
		splitted = sentence.split('\n')
		for line in splitted:
			s = line.split()
			for word in s:
				#print word
				if word in d:
					#print d[word]
					sentScore += d[word]
					numHits += 1.0
		if sentScore > 0:
			sentScore = sentScore/numHits
		scores.append(sentScore)
	return scores


def trainTFID(corpus):
	"trains tf-idf model on corpus"
	vect = TfidfVectorizer(min_df=0)

	# turns corpus into desired form
	"""
	corp = []
	for document in corpus:
		for paragraph in document:
			for sentence in paragraph:
				corp.append(" ".join(sentence))

	"""
	corp = []
	for doc in corpus:
		dc = " ".join(doc)
		corp.append(dc)


	vect.fit(corp)
	corpus_tf_idf = vect.transform(corp)

	# dictionary of all terms and their tf-idf
	return dict(zip(vect.get_feature_names(), corpus_tf_idf.data))

def getKeyWords(corp, toScore):
	"""extract values for how likely a word is to be a key word
	"""
	newDict = {}
	for word in toScore:
		if word in corp:
			score = corp[word] - toScore[word]
			#print word, score
		else:
			score = 0.0
		newDict[word] = score
	return newDict

def scoreByKeyWords(keywords, toScore):
	"""score sentences using
	the key word socres """
	scores = []
	for sentence in toScore:
		sentScore = 0.0
		numHits = 0.0
		splitted = sentence.split('\n')
		for line in splitted:
			s = line.split()
			for word in s:
				if word in keywords:
					sentScore += keywords[word]
					numHits += 1.0
		if sentScore > 0:
			sentScore = sentScore/numHits
		scores.append(sentScore)
	return scores

def getSummary(scores, toSummarize, numSents):
	"""get summary based on highest scores"""
	#print scores, toSummarize
	summary = []
	for i in range(numSents):
		arg = np.argmax(scores)
		scores[arg] = -float("inf")
		summary += [toSummarize[arg]]
	return " ".join(summary)

def printDoc(doc):
	"""print out original document"""
	print " ".join(doc)

def flattenList(l):
	"""Flatten documents into single list"""
	flat = []
	for paragraph in l:
		for sentence in paragraph:
			flat.append(sentence)
	return flat

def getRandomScores(doc):
	""" Random scoring for comparisons"""
	scores = []
	for sentence in doc:
		scores += [random.random()]
	return scores


if __name__ == '__main__':
	inputFile = sys.argv[1]
	document = st.writeSentences(inputFile)

	randomScores = getRandomScores(document)

	corpus = []
	for filename in os.listdir('corpus/fulltext'):
		if ('.xml' in filename):
			corpus += [st.writeSentences(filename)]

	corp_tf_idf = trainTFID(corpus)
	example_tf_idf = trainTFID([document])
	keywords = getKeyWords(corp_tf_idf, example_tf_idf)
	#print corp_tf_idf
	tfidf_ranks = tfidfRank(corp_tf_idf, document)
	keywords_scores = scoreByKeyWords(keywords, document)

	print keywords_scores, tfidf_ranks

	print "ORIGINAL DOCUMENT"
	print
	printDoc(document)
	print
	print "RANDOM SUMMARY"
	print
	print getSummary(randomScores, document, 2)
	print
	print "KEYWORDS SUMMARY"
	print
	print getSummary(keywords_scores, document, 2)
	print
	print "TFIDF SUMMARY"
	print
	print getSummary(tfidf_ranks, document, 2)

