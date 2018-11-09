import re
import nltk
import nltk.treetransforms
import functools
import pprint
from nltk.draw.tree import draw_trees
import numpy as np
import sys
import time
class PCKY:
    def __init__(self, input_grammar):
        #Initializing grammar with the given input grammar
        self.grammar = input_grammar
        self.all_productions = {}

        if not grammar.is_chomsky_normal_form():
            print("Grammar is not in CNF, cannot initialize")
            return

        #Create a dictionary of all RHS NT pairs as key and the productions as value
        for prod in grammar.productions():
            if len(prod.rhs()) == 2:
                tuple = prod.rhs()
                if tuple not in self.all_productions:
                    self.all_productions[tuple] = [prod]
                else:
                    self.all_productions[tuple].append(prod)

        print("Grammar initialization complete")

    def compare_tree(cls, tree1, tree2): #compares the probabilities of the trees
        if tree1.prob() > tree2.prob():
            return -1
        elif tree1.prob() == tree2.prob():
            return 0
        else:
            return 1

    compare_tree = classmethod(compare_tree)

    def parse_sentence(self,sentence):

        if not grammar.is_chomsky_normal_form():
            return

        #generate tokens from a given sentence
        #tokens = nltk.tokenize.wordpunct_tokenize(sentence)
        tokens = str.split(sentence)
        N = len(tokens)

        #create a parse table of size [# of tokens +1][# of tokens +1]
        self.parse_table = np.zeros((N+1,N+1), dtype=object)

        for j in range(1,N+1):
        #going from left to right
            prod_list = []

            for prod in self.grammar.productions():
                if prod.rhs()[0] == tokens[j-1]:
                    #if the  rhs terminal of the production matches with the token[j-1], append it to the prod list
                    prod_list.append(prod)

            trees =[]
            for prod in prod_list:
                trees.append(nltk.ProbabilisticTree(prod.lhs().symbol(), [prod.rhs()], prob=prod.logprob()))

            self.parse_table[j-1][j] = trees

            for i in range(j - 2, -1, -1):
            #going from down to up
                t ={}
                for k in range(i+1,j):
                #checking all combinatoins of the split, for mathing non terminal combinations
                    left_candidates = self.parse_table[i][k] #get all LHS candidates
                    right_candidates = self.parse_table[k][j] #get all RHS candidates

                    if left_candidates and right_candidates:

                        for l_node in left_candidates:
                            for r_node in right_candidates:
                                parent_node = (nltk.grammar.Nonterminal(str(l_node.label())),nltk.grammar.Nonterminal(str(r_node.label())))
                                if parent_node in self.all_productions:
                                    for prod in self.all_productions[parent_node]:
                                        prob = l_node.prob() + r_node.prob()
                                        prob = prob + prod.logprob() #probability of new tree is the sum of log probabilities of left and right child tree and the log probability of the rule itself
                                        et = None
                                        if prod.lhs() in t:
                                            et = t[prod.lhs()]
                                        if et == None or prob > et.prob(): #update the existing tree in dictionary only if the new prob for the given parent node is greater than the existing value
                                            t[prod.lhs()] = nltk.ProbabilisticTree(prod.lhs(), [l_node, r_node], prob = prob)#generate the tree iteratively

                treesToKeep = t.values()
                treesToKeep = sorted(treesToKeep, key = functools.cmp_to_key(PCKY.compare_tree)) # Sort trees, with highest probabilities first
                self.parse_table[i][j] = treesToKeep


    def get_all_parses(self, sentence):
        #for the input sentense, use the parse_table to find all the parse trees that apply
        self.parse_sentence(sentence)
        tokens = nltk.tokenize.wordpunct_tokenize(sentence)
        tokens = str.split(sentence)
        N = len(tokens)
        parse_start = self.parse_table[0][N] #find the start/topmost cell in the parse_table
        result = []
        if parse_start != None:
            for t in parse_start:
                if t.label() == self.grammar.start(): #if the start symbol of the grammar matches the start symbol of the tree, append the tree
                    result.append(t)
        return result


if __name__ == '__main__':


    if (len(sys.argv) > 2):
        input_PCFG_file = sys.argv[1]
        test_sentence_filename = sys.argv[2]
        output_parse_filename = sys.argv[3]
    else:
        print("Insufficient arguments passed")


    sentences = open(test_sentence_filename,'r')
    grammar = nltk.data.load(input_PCFG_file, 'pcfg')


    parser = PCKY(grammar)

    count_no_parse_sentences = 0

    start_time = time.clock()
    with open(output_parse_filename,'w') as write_file:
        for sentence in sentences:
            parses = parser.get_all_parses(sentence) #get all parses for the sentences
            if len(parses) == 0:
                count_no_parse_sentences +=1
                write_file.write("" + '\n')
            else:
                s = parses[0].pformat( parens='()', quotes=False) #print the best parse - parses[0] since these are presorted
                s = s.split('\n')
                s = [item.strip() for item in s]
                flat_tree = ' '.join(s)
                write_file.write(flat_tree+'\n')
    print("# sentences for which parse not found: "+ str(count_no_parse_sentences))
    print("Top parses for sentences written to file - complete")
    print('Total time taken in seconds: ' + str((time.clock() - start_time)))