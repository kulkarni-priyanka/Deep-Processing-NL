import nltk
import sys
from nltk.tree import Tree
from nltk import treetransforms
from copy import deepcopy

def generate_pcfg(grammar,pcfg_grammar_file):
    count_overall = {}
    count_rule = {}
    probability_rule ={}

    # Total count of non terminals
    for prod in grammar.productions():
        if prod.lhs() not in count_overall:
            count_overall[prod.lhs()] = 1
        else:
            count_overall[prod.lhs()] += 1
    # Total count for each rule
    for prod in grammar.productions():
        if prod not in count_rule:
            count_rule[prod] = 1
        else:
            count_rule[prod] += 1
    # probability of reach rule
    for entry in count_rule:
        probability_rule[entry] = float(count_rule[entry])/count_overall[entry.lhs()]
    # write to file
    with open(pcfg_grammar_file,'w') as pcfg_op:
        pcfg_op.write('%start ' + str(grammar.start()) + '\n')
        for key,val in probability_rule.items():
            pcfg_op.write(str(key)+" ["+str(val)+"]\n")

    print("PCFG grammar creation complete")

def generate_all_productions(records,productions_file):
    productions = []
    start_symbol = ""
    for line in records:

        if len(line) > 2:
            tree = Tree.fromstring(line) #generate tree from string
            start_symbol = tree.label()#get start symbol
            productions = productions + tree.productions()
            parentTree = deepcopy(tree)
            treetransforms.chomsky_normal_form(parentTree, horzMarkov=2, vertMarkov=1) #parent annotation (one level) and horizontal smoothing of order two
            productions = productions + parentTree.productions() #get tree productions and append to list

    with open(productions_file,'w') as prod_write:  #write productions to a file
        prod_write.write('%start ' + str(start_symbol) + '\n')  # explicitly set the start marker
        for prod in productions:
            prod_write.write(str(prod)+"\n")
    print("Production rules generated")



if __name__ == '__main__':

    if (len(sys.argv) >=2):
        treebank_filename = sys.argv[1]
        pcfg_grammar_file = sys.argv[2]
    else:
        print("Incorrect number of arguments")


    productions_file ="all_generated_productions_improved.cfg"
    treebank_records = open(treebank_filename, "r")
    generate_all_productions(treebank_records,productions_file)
    grammar = nltk.data.load(productions_file,'cfg')

    generate_pcfg(grammar,pcfg_grammar_file)











