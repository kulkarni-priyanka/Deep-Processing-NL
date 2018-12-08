# Mike Roylance - roylance@uw.edu
import nltk

def buildTreesFromSentences(firstSentence, secondSentence, parser):
	# tokenize the sentences
	firstTokenizedSentence = nltk.word_tokenize(firstSentence)
	secondTokenizedSentence = nltk.word_tokenize(secondSentence)

	# parse them
	firstTrees = parser.nbest_parse(firstTokenizedSentence)
	secondTrees = parser.nbest_parse(secondTokenizedSentence)

	if len(firstTrees) > 0 and len(secondTrees) > 0:
		return (firstTrees[0], secondTrees[0])

	return (None, None)