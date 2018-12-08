# Mike Roylance - roylance@uw.edu
import sys
import nltk


def buildTreesFromSentences(firstSentence, secondSentence, parser):
    # tokenize the sentences
    firstTokenizedSentence = nltk.word_tokenize(firstSentence)
    secondTokenizedSentence = nltk.word_tokenize(secondSentence)

    # parse them
    firstTrees = parser.parse(firstTokenizedSentence)
    secondTrees = parser.parse(secondTokenizedSentence)

    if len(firstTrees) > 0 and len(secondTrees) > 0:
        return (firstTrees[0], secondTrees[0])

    return (None, None)

def main():
    # arg variables
    grammarFile = '../data/grammar.cfg'
    sentenceFile = '../data/coref_sentences.txt'
    output_filename = '../data/hw9_output.txt'

    # create grammar
    grammar = nltk.data.load(grammarFile)
    parser = nltk.parse.EarleyChartParser(grammar)

    # read in the sentences
    sentenceList = []
    sentenceFileStream = open(sentenceFile)
    sentence = sentenceFileStream.readline()

    while sentence:
        # first, strip sentence so we can detect empty string
        strippedSentence = sentence.strip()
        # add if length is 0
        if len(sentenceList) == 0 and len(strippedSentence) > 0:
            sentenceList.append(strippedSentence)
        # process if length is 1
        elif len(sentenceList) == 1 and len(strippedSentence) > 0:
            trees = buildTreesFromSentences(sentenceList[0], strippedSentence, parser)

        # read new sentence
        sentence = sentenceFileStream.readline()


if __name__ == '__main__':
    main()