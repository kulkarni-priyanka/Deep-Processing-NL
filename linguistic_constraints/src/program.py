import nltk
import sys

if __name__ == "__main__":

    text = nltk.word_tokenize("Did Mary put the book on the shelf")
    print(nltk.pos_tag(text))

    if (len(sys.argv) >=2):
        fcfg_grammar_file = sys.argv[1]
        sentences_filename = sys.argv[2]
        output_filename = sys.argv[3]
    else:
        fcfg_grammar_file = "../data/hw5_feature_grammar.fcfg"
        sentences_filename = "../data/sent.txt"
        output_filename = "../data/ex_sentences_output.txt"
        #print("Incorrect number of arguments")

    # Loading grammar
    grammar = nltk.data.load(fcfg_grammar_file,'fcfg')

    # Setting the parser
    parser = nltk.parse.FeatureEarleyChartParser(grammar)

    # Opening the sentence file that needs to be parsed
    sentence_file = open(sentences_filename)

    sentences = sentence_file.read().split('\n')
    sentence_count = 0
    overall_parse_count = 0

    with open(output_filename, 'w') as op_file:
        for sentence in sentences:
            if (len(sentence) > 0):
                # increment number of sentences
                sentence_count += 1
                sentence = sentence.strip()  # strip extra whitepaces if any

                # split_sentence = re.findall(r"[\w']+|[.,!?;]", sentence) # splitting using regex

                split_sentence = nltk.word_tokenize(sentence)  # split sentence
                trees = parser.parse(split_sentence)  # parse the sentence
                parse_count = 0

                tree_strings = ""


                if not trees:
                    tree_strings = ['()']
                else:
                    first = True
                    for t in trees:
                        if first:
                            tree_strings = t.pformat(margin=sys.maxsize)
                            first = False

                op_file.write(tree_strings + '\n')

                # increment overall parse count
                overall_parse_count += parse_count


        # Compute average and print the average parses per sentence
        #op_file.write('Average parses per sentence: ' + str(overall_parse_count / float(sentence_count)))
        print('Parsing complete')
        #print('Average parses per sentence: ' + str(round((overall_parse_count / float(sentence_count)), 3)))