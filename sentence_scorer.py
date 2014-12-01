# Kathleen Schaefer and Johnathan Ashley

#pip install -U numpy scipy scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer #
import numpy as np

sent1 = "This is the first sentence of paragraph one ."
sent2 = "This is the last sentence of paragraph one ."
sent3 = "This is a highly important sentence with highly important facts ."
example = [[sent1.split(), sent2.split()],[sent3.split()]]


corpSent1 = "Look at this wonderful with corpus of words "
corpSent2 = "Isn ' t it so of wonderful ?"
corpSent3 = "Here is another of of corpus with common words a with of to a."
corpSent4 = "This one has two paragraph with of to the of this a."
corpus = [[[corpSent1.split(), corpSent2.split()]],
	[[corpSent3.split()],[corpSent4.split()]]]

def tfidfRank(d, toScore):
	""" calculates the average tf-idf score of every sentence
	in the example for the given corpus """

	scores = []
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
	return scores


def trainTFID(corpus):
	"trains tf-idf model on corpus"
	vect = TfidfVectorizer(min_df=1)

	# turns corpus into desired form
	corp = []
	for document in corpus:
		for paragraph in document:
			for sentence in paragraph:
				corp.append(" ".join(sentence))

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
		else:
			score = 0.0
		newDict[word] = score
	return newDict

def scoreByKeyWords(keywords, toScore):
	"""score sentences using
	the key word socres """
	scores = []
	for paragraph in toScore:
		parScores = []
		for sentence in paragraph:
			sentScore = 0.0
			numHits = 0.0
			for word in sentence:
				if word in keywords:
					sentScore += keywords[word]
					numHits += 1.0
			if sentScore > 0:
				sentScore = sentScore/numHits
			parScores.append(sentScore)
		scores.append(parScores)
	return scores

def getSummary(scores, toSummarize, numSents):
	"""get summary based on highest scores"""
	summary = []
	for i in range(numSents):
		arg = np.argmax(scores)
		scores[arg] = -float("inf")
		summary += toSummarize[arg]
	return " ".join(summary)

def flattenList(l):
	"""Flatten documents into single list"""
	flat = []
	for paragraph in l:
		for sentence in paragraph:
			flat.append(sentence)
	return flat


if __name__ == '__main__':

	corp_tf_idf = trainTFID(corpus)
	example_tf_idf = trainTFID([example])
	keywords = getKeyWords(corp_tf_idf, example_tf_idf)
	tfidf_ranks = tfidfRank(corp_tf_idf, example)
	keywords_scores = scoreByKeyWords(keywords, example)


	print getSummary(flattenList(tfidf_ranks), flattenList(example), 2)

