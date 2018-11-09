'''
Assignment 1
Load the grammar
Build a parser for the grammar using nltk.parse.EarleyChartParser
Read in the example sentences
For each example sentence, output to a file:
The sentence itself
The simple bracketed structure parse(s), and
the number of parses for that sentence.
Finally, print the average number of parses per sentence obtained by the grammar.
'''

import nltk
import re
import sys

#Taking grammar, sentences file and output file names as parameters
'''
grammar_file = sys.argv[1]
sentences_file = sys.argv[2]
output_file = sys.argv[3]
'''

grammar_file = '../atis_v2.cfg'
sentences_file = '../data/sentences.txt'
output_file = 'atis_cfg_v2.txt'

#Opening the sentence file that needs to be parsed
sentence_file = open(sentences_file,'r',encoding='utf-8')


#Loading grammar
grammar = nltk.data.load(grammar_file)

#Setting the parser
parser = nltk.parse.EarleyChartParser(grammar)



sentences = sentence_file.read().split('\n')
sentence_count = 0
overall_parse_count = 0

with open(output_file,'w') as op_file:
    for sentence in sentences:
        if(len(sentence)>0):
            #increment number of sentences
            sentence_count +=1
            op_file.write(sentence)
            op_file.write('\n')
            sentence = sentence.strip() #strip extra whitepaces if any

            #split_sentence = re.findall(r"[\w']+|[.,!?;]", sentence) # splitting using regex

            split_sentence = nltk.word_tokenize(sentence) #split sentence
            trees = parser.parse(split_sentence) #parse the sentence
            parse_count = 0

            for tree in trees:
                #count the number of trees/parses
                parse_count +=1
                op_file.write(str(tree))
                op_file.write('\n')

            op_file.write("Number of parses: "+str(parse_count))
            op_file.write('\n')

            #increment overall parse count
            overall_parse_count += parse_count
            op_file.write('\n')

    #Compute average and print the average parses per sentence
    op_file.write('Average parses per sentence: '+str(overall_parse_count/sentence_count))
    print('Parsing complete')
    print('Average parses per sentence: '+str(round((overall_parse_count/sentence_count),3)))