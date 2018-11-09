import nltk
from nltk.tree import *
import sys
import re
import os
import src.cfgToCnfBuilder
import src.productionBuilder

def main():
	# arg variables
	grammarFile = '../data/atis.cfg'

	input_stream = open(grammarFile)
	grammarText = input_stream.read()
	input_stream.close()

	# convert the original grammar to CNF
	cfgBuilder = src.cfgToCnfBuilder.CfgToCnfBuilder(grammarText)
	cfgBuilder.build()

	# load in the CNF grammar
	cnfGrammar = cfgBuilder.getFinalProductions()

	# print out the rules of the converted grammar to a file named grammar.cnf
	fileName = 'atis-grammar.cfg'
	try:
		os.remove(fileName)
	except OSError:
		pass

	target = open(fileName, 'w')

	for production in cnfGrammar:
		target.write(str(production))
		target.write('\n')

main()