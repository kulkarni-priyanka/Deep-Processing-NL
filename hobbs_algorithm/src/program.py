import sys
import time
import nltk

if __name__ == "__main__":

    if (len(sys.argv) >=2):
        input_grammar_filename = sys.argv[1]
        test_sentence_filename = sys.argv[2]
        output_filename = sys.argv[3]
    else:
        input_grammar_filename = '../data/grammar.cfg'
        test_sentence_filename = '../data/coref_sentences.txt'
        output_filename = '../data/test_output.txt'
        print("Incorrect number of arguments")

    start = time.clock()

    grammar = nltk.data.load(input_grammar_filename,'fcfg')

    print("Grammar loaded")

    # Setting the parser
    parser = nltk.parse.FeatureEarleyChartParser(grammar)

    # Opening the sentence file that needs to be parsed
    sentence_file = open(test_sentence_filename)

    sentences = sentence_file.read().split('\n')
    sentence_count = 0

    print("Beginning the parsing process")

    with open(output_filename, 'w') as op_file:
        for sentence in sentences:
            if (len(sentence) > 0):
                # increment number of sentences
                sentence_count +=1

                sentence = sentence.strip()  # strip extra whitepaces if any

                split_sentence = nltk.word_tokenize(sentence)  # split sentence
                trees = parser.parse_one(split_sentence)  # parse the sentence

                tree_strings = [""] if not trees else [trees.pformat(margin=sys.maxsize) ]
                op_file.write('\n'.join(tree_strings) + '\n')

        print(str(sentence_count)+" sentences parsed ")



    print('Parsing complete')
    print("Time taken :" + str(time.clock() - start))
