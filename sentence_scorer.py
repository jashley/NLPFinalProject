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

def tfidfRank(d, toScore):
	""" calculates the average tf-idf score of every sentence
	in the example for the given corpus """

	scores = []

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
		summary += [st.cleanSentenceKeepPunctuation(toSummarize[arg])]
	return "\n".join(summary)

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
				#print word, "too many docs", docVal/numDocs
				scores[word] = 0.0
			#elif totalVal/docVal > freq:
				#print word, "too infrequent", totalVal/docVal
				#scores[word] = 0.0
			#elif freq/numWords < .001:
				#print word, "uncommon word"
				#scores[word] = 0.0
			elif getAlphaRatio(word) < .75:
				#print word, "non-alpha"
				scores[word] = 0.0
			else:
				score = freq/docVal
				scores[word] = score
				#print word, score
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
	""" normalize distribution of scores"""
	total = sum(scores)
	return [n/total for n in scores]


if __name__ == '__main__':


	corpus = []
	numDocs = 0.0
	for filename in os.listdir('corpus/fulltext'):
		if ('.xml' in filename):
			numDocs += 1.0
			#corpus += [st.writeSentences(filename)]
			corpus += [st.getCleanedSentences(filename)]
	print "done loading and cleaning files"
	totalCorpus, docCorpus = countWords(corpus)
	print "done counting corpus words"

	#inputFile = sys.argv[1]
	for inputFile in os.listdir('corpus/fulltext'):
	
		document = st.writeSentences(inputFile)
		raw_document = st.writeSentences(inputFile, cleaned = False)

		random_scores = getRandomScores(document)
		position_scores = positionalScores(document)

	
		totalDoc, x = countWords([document])
		scoreWords = scoreWordsInDoc(numDocs, totalCorpus, docCorpus, totalDoc)
		keywords_scores = scoreByKeyWords(scoreWords, document)
		combined_scores = [x + y for x, y in zip(keywords_scores, position_scores)]
		#summary_filename = sys.argv[1][:-4]+".txt"
		summary_filename = inputFile[:-4]
		
		# write files
		f = open('random_summary/'+summary_filename+".random.system", 'w+')
		summary = getSummary(random_scores, raw_document, 3)
		f.write(summary)
		f.close()
		f = open('keyword_summary/'+summary_filename+".keyword.system", 'w+')
		summary = getSummary(keywords_scores, raw_document, 3)
		f.write(summary)
		f.close()
		f = open('position_summary/'+summary_filename+".positional.system", 'w+')
		summary = getSummary(position_scores, raw_document, 3)
		f.write(summary)
		f.close()
		f = open('combined_summary/'+summary_filename+".combined.system", 'w+')
		summary = getSummary(combined_scores, raw_document, 3)
		f.write(summary)
		f.close()
		print "wrote " + summary_filename

	#corp_tf_idf = trainTFID(corpus)
	#example_tf_idf = trainTFID([document])
	#keywords = getKeyWords(corp_tf_idf, example_tf_idf)
	#print corp_tf_idf
	#tfidf_ranks = tfidfRank(corp_tf_idf, document)
	#keywords_scores = scoreByKeyWords(keywords, document)
	





