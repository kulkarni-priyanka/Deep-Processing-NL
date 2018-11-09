import nltk
import sys
from nltk.sem import cooper_storage as cs

if __name__ == "__main__":

    if (len(sys.argv) >=2):
        fcfg_grammar_file = sys.argv[1]
        sentences_filename = sys.argv[2]
        output_filename = sys.argv[3]
    else:
        fcfg_grammar_file = "../data/hw6_semantic_grammar.fcfg"
        sentences_filename = "../data/sentences.txt"
        output_filename = "../data/hw6_output.txt"
        #print("Incorrect number of arguments")

    # Loading grammar
    grammar = nltk.data.load(fcfg_grammar_file,'fcfg')

    print("Grammar loaded")

    # Setting the parser
    parser = nltk.parse.FeatureEarleyChartParser(grammar)

    # Opening the sentence file that needs to be parsed
    sentence_file = open(sentences_filename)

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
                tree = parser.parse_one(split_sentence)  # parse the sentence
                op_file.write(sentence+"\n")
                if not tree:
                    op_file.write("\n")
                else:
                    semrep = tree.label()['SEM'].simplify()
                    op_file.write(str(semrep)+ '\n')

        print(str(sentence_count)+" sentences parsed ")
        print('Parsing complete')
