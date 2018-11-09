import nltk
import sys
import time
import numpy as np


class CKY:
    def __init__(self, input_grammar):

        self.grammar = input_grammar
        self.all_productions = {}

        for prod in grammar.productions():
            if len(prod.rhs()) == 2:
                tuple = prod.rhs()
                if tuple not in self.all_productions:
                    self.all_productions[tuple] = [prod]
                else:
                    self.all_productions[tuple].append(prod)

    def parse_sentence(self,sentence):
        tokens = nltk.tokenize.wordpunct_tokenize(sentence)
        N = len(tokens)
        self.parse_table = np.zeros((N+1,N+1), dtype=object)

        for j in range(1,N+1):
            prod_list = []

            for prod in self.grammar.productions():
                if prod.rhs()[0] == tokens[j-1]:
                    prod_list.append(prod)

            trees =[]
            for prod in prod_list:
                trees.append(nltk.Tree(prod.lhs(), [prod.rhs()]))

            self.parse_table[j-1][j] = trees

            for i in range(j - 2, -1, -1):
                t =[]
                for k in range(i+1,j):
                    left_candidates = self.parse_table[i][k]
                    right_candidates = self.parse_table[k][j]

                    if left_candidates and right_candidates:

                        for l_node in left_candidates:
                            for r_node in right_candidates:
                                parent_node = (l_node.label(),r_node.label())

                                if parent_node in self.all_productions:
                                    for prod in self.all_productions[parent_node]:
                                        t.append(nltk.Tree(prod.lhs(), [l_node, r_node]))

                self.parse_table[i][j] = t

    def get_all_parses(self, sentence):
        """
        Given a sentence, generates a parse chart for it,
        and returns a list of string representations of all
        of its parses.
        """
        self.parse_sentence(sentence)
        tokens = nltk.tokenize.wordpunct_tokenize(sentence)
        N = len(tokens)
        parse_start = self.parse_table[0][N]
        result = []
        if parse_start != None:
            for t in parse_start:
                if t.label() == self.grammar.start():
                    result.append(t)
        return result

if __name__ == "__main__":

    start_time = time.clock()

    if (len(sys.argv) >= 2 ):
        grammar_file = sys.argv[1]
        test_sentences_file = sys.argv[2]
        output_file = sys.argv[3]

    else:
        test_sentences_file = "../data/sentences.txt"
        grammar_file = "../data/grammar_cnf.cfg"
        output_file = "output.txt"


    grammar = nltk.data.load(grammar_file)
    parser = CKY(grammar)
    sentences_open = open(test_sentences_file, "r", encoding='utf-8')
    sentences = sentences_open.read().split('\n')
    overall_parse_count = 0
    sentence_count = 0

    with open(output_file, 'w') as op_file:
        for sentence in sentences:
            if(len(sentence)>0):
                sentence_count += 1
                op_file.write(sentence)
                op_file.write('\n')
                trees = parser.get_all_parses(sentence)
                parse_count = 0
                for tree in trees:
                    # count the number of trees/parses
                    parse_count += 1
                    op_file.write(str(tree))
                    op_file.write('\n')

                overall_parse_count += parse_count

                op_file.write("Number of parses: " + str(parse_count)+"\n")
                op_file.write('\n')

    print('Total time taken in seconds: ' + str((time.clock() - start_time)))
    print('Parsing complete')
    print('Average parses per sentence: ' + str(round((overall_parse_count / sentence_count), 3)))
