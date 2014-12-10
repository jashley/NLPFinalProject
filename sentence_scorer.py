# Kathleen Schaefer and Johnathan Ashley

# ideas from
# http://courses.ischool.berkeley.edu/i256/f06/papers/edmonson69.pdf

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
	return normalizeScores(scores)

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

def getRandomScores(doc):
	""" Random scoring for comparisons"""
	scores = []
	for sentence in doc:
		scores += [random.random()]
	return scores

def countWords(corpus):
	""" counts occurences of words in a corpus """
	docCounts = {}
	totalCounts = {}
	for doc in corpus:
		docDic = {}
		for sentence in doc:
			splitted = sentence.split('\n')
			for line in splitted:
				s = line.split()
				for word in s:
					docDic[word] = 1.0
					if word in totalCounts:
						totalCounts[word] += 1.0
					else:
						totalCounts[word] = 1.0
		for word in docDic:
			if word in docCounts:
				docCounts[word] += 1.0
			else:
				docCounts[word] = 1.0
	return totalCounts, docCounts


def scoreWordsInDoc(numDocs, totalCounts, docCounts, totalDoc):
	""" scores words based on how likely they are to 
	be key words """
	scores = {}
	numWords = len(totalDoc)
	for word in totalDoc:
		freq = totalDoc[word]
		if word in docCounts:
			totalVal = totalCounts[word]
			docVal = docCounts[word]
			if docVal/numDocs >= .5:
				print word, "too many docs", docVal/numDocs
				scores[word] = 0.0
			#elif totalVal/docVal > freq:
				#print word, "too infrequent", totalVal/docVal
				#scores[word] = 0.0
			elif freq/numWords < .001:
				print word, "uncommon word"
				scores[word] = 0.0
			elif getAlphaRatio(word) < .75:
				print word, "non-alpha"
				scores[word] = 0.0
			else:
				score = freq/docVal
				scores[word] = score
				print word, score
	return scores

def getAlphaRatio(word):
	""" calculates the ratio of alpha to nonalpha scores
	in the word """
	length = len(word)
	alpha = 0.0
	for letter in word:
		if letter.isalpha():
			alpha += 1.0
	#print "ALPHA", word, alpha/length
	return alpha/length

def cleanSentence(text):
    sent = []
    exclude = set(string.punctuation)
    for word in text.split():
        #may want to do better things
        # with punctuation
        s = ''.join(ch.lower() for ch in word if ch not in exclude)
        sent += [s]
    return " ".join(sent)

def positionalScores(toScore):
	""" score sentences higher if they are close
	to the beginning or end of the document"""
	midpoint = len(toScore)/2
	scores = []
	for i in range(len(toScore)):
		score = float(abs(midpoint - i))**10
		scores += [score]
	return normalizeScores(scores)

def normalizeScores(scores):
	total = sum(scores)
	return [n/total for n in scores]


if __name__ == '__main__':
	inputFile = sys.argv[1]
	document = st.writeSentences(inputFile)
	raw_document = st.writeSentences(inputFile, cleaned = False)

	randomScores = getRandomScores(document)
	position_scores = positionalScores(document)

	corpus = []
	numDocs = 0.0
	for filename in os.listdir('corpus/fulltext'):
		if ('.xml' in filename):
			numDocs += 1.0
			corpus += [st.writeSentences(filename)]
	print "done loading and cleaning files"

	totalCorpus, docCorpus = countWords(corpus)
	totalDoc, x = countWords([document])
	scoreWords = scoreWordsInDoc(numDocs, totalCorpus, docCorpus, totalDoc)

	#corp_tf_idf = trainTFID(corpus)
	#example_tf_idf = trainTFID([document])
	#keywords = getKeyWords(corp_tf_idf, example_tf_idf)
	#print corp_tf_idf
	#tfidf_ranks = tfidfRank(corp_tf_idf, document)
	#keywords_scores = scoreByKeyWords(keywords, document)
	keywords_scores = scoreByKeyWords(scoreWords, document)

	combined_scores = [x + y for x, y in zip(keywords_scores, position_scores)]

	#print keywords_scores, tfidf_ranks

	#print "ORIGINAL DOCUMENT"
	#print
	#printDoc(document)
	#print
	#print "RANDOM SUMMARY"
	#print

	#print getSummary(randomScores, raw_document, 2)
	#print
	#print "KEYWORDS SUMMARY"
	#print keywords_scores
	#print getSummary(keywords_scores, raw_document, 3)
	f = open('keyword_summary/'+sys.argv[1], 'w+')
	f.write(getSummary(keywords_scores, raw_document, 3))
	f.close()
	print
	#print "POSITION SUMMARY"
	#print position_scores
	f = open('position_summary/'+sys.argv[1], 'w+')
	f.write(getSummary(position_scores, raw_document, 3))
	f.close()
	#print getSummary(position_scores, raw_document, 3)
	print
	#print "COMBINED SUMMARY"
	#print combined_scores
	#print getSummary(combined_scores, raw_document, 3)
	f = open('combined_summary/'+sys.argv[1], 'w+')
	f.write(getSummary(combined_scores, raw_document, 3))
	f.close()
	#print
	#print "TFIDF SUMMARY"
	#print
	#print getSummary(tfidf_ranks, document, 2)

