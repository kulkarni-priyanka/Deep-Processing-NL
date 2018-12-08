import sys
import time
import nltk
from nltk.grammar import Production, Nonterminal

def get_all_pronouns(tree):
    pronouns_available =[]
    for production in tree.productions():
        if (production._lhs== Nonterminal('PRP') or production._lhs== Nonterminal('PossPro')) and type(production._rhs[0]) is str :
            pronouns_available.append(production._rhs[0])
    return pronouns_available


if __name__ == "__main__":

    if (len(sys.argv) >=2):
        input_grammar_filename = sys.argv[1]
        test_sentence_filename = sys.argv[2]
        output_filename = sys.argv[3]
    else:
        input_grammar_filename = '../data/grammar.cfg'
        test_sentence_filename = '../data/coref_sentences.txt'
        output_filename = '../data/hw9_output.txt'
        print("Incorrect number of arguments")

    start = time.clock()

    grammar = nltk.data.load(input_grammar_filename,'cfg')

    print("Grammar loaded")

    # Setting the parser
    parser = nltk.parse.EarleyChartParser(grammar) #

    # Opening the sentence file that needs to be parsed
    sentence_file = open(test_sentence_filename)

    sentences = sentence_file.read().split('\n')
    sentence_buffer= []

    print("Beginning the parsing process")

    with open(output_filename, 'w') as op_file:
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence_buffer) < 3 and len(sentence) >1:
                sentence_buffer.append(sentence)
                if len(sentence_buffer) == 2:
                    sent1 = nltk.word_tokenize(sentence_buffer[0])
                    sent2 = nltk.word_tokenize(sentence_buffer[1])
                    trees_sent1 = parser.parse_one(sent1)
                    trees_sent2 = parser.parse_one(sent2)
                    all_pronouns = get_all_pronouns(trees_sent2)
                    for pronoun in all_pronouns:
                        op_file.write(pronoun+'\t')
                        op_file.write(trees_sent1.pformat(margin=float("inf"))+' ')
                        op_file.write(trees_sent2.pformat(margin=float("inf")))
                        op_file.write('\n')
                    print(all_pronouns)
                    sentence_buffer = []

    print('Parsing complete')
    print("Time taken :" + str(time.clock() - start))
