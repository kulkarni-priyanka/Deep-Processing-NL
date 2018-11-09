import nltk
import sys
from nltk.tree import Tree
import operator
import math
from collections import OrderedDict
from operator import itemgetter

def get_terminal_frequency_and_count(grammar,sampling_rate, oov_productions_file):
    word_freq= {}

    for prod in grammar.productions():
        if nltk.grammar.is_terminal(prod.rhs()[0]):
            if prod.rhs() not in word_freq:
                word_freq[prod.rhs()] = 1
            else:
                word_freq[prod.rhs()] +=1

    total_terminal_count = sum(v for v in word_freq.values())
    replacement_count = math.floor(total_terminal_count*sampling_rate)

    word_freq = OrderedDict(sorted(word_freq.items(), key=itemgetter(1)))

    replace_words =[]
    #Identify words with frequency < 2 and replace by UNK
    print('Replacing low frequncy unigrams by UNK')

    #https://en.wikipedia.org/wiki/Part_of_speech#Open_and_closed_classes


    closed_class = ['PRP', 'PRP' , 'WP', 'DT', 'PDT', 'WDT','CC', 'IN' , 'TO']

    #sort in ascending order

    init_count = 0

    for elements in list(word_freq):
        if init_count < replacement_count:
            word_tag = nltk.pos_tag([elements[0]])[0][1]
            if word_tag not in closed_class:
                replace_words.append(elements)
                popped_val = word_freq.pop(elements)
                init_count +=  popped_val
                if 'UNK' not in word_freq.keys():
                    word_freq[('UNK')] = popped_val
                else:
                    word_freq[('UNK')] += popped_val

    with open(oov_productions_file,'w') as prod_write:
        prod_write.write('%start ' + str(grammar.start().symbol()) + '\n')  # explicitly set the start marker
        for prod in grammar.productions():
            if prod.rhs() in replace_words:
                prod = nltk.grammar.Production(prod.lhs(), tuple(['UNK',]))
            prod_write.write(str(prod)+"\n") #write productions to a file
    print("Production rules generated")
    print("Dictionary construction complete")


def generate_pcfg(grammar,pcfg_grammar_file):
    count_overall = {}
    count_rule = {}
    probability_rule =  {}

    #Total count of non terminals
    for prod in grammar.productions():
        if prod.lhs() not in count_overall:
            count_overall[prod.lhs()] = 1
        else:
            count_overall[prod.lhs()] += 1
    #Total count for ewach rule
    for prod in grammar.productions():
        if prod not in count_rule:
            count_rule[prod] = 1
        else:
            count_rule[prod] += 1
    #probability of reach rule
    for entry in count_rule:
        probability_rule[entry] = float(count_rule[entry])/count_overall[entry.lhs()]

    #write to file
    with open(pcfg_grammar_file,'w') as pcfg_op:
        pcfg_op.write('%start ' + str(grammar.start()) + '\n')
        for key,val in probability_rule.items():
            pcfg_op.write(str(key)+" ["+str(val)+"]\n")

    print("PCFG grammar creation complete")

def generate_all_productions(records,productions_file):
    productions = []
    for line in records:
        start_symbol = ""
        if len(line) > 2:
            tree = Tree.fromstring(line) #generate tree from string
            start_symbol = tree.label() #get start sybol
            productions = productions + tree.productions() #get tree productions and append to list

    with open(productions_file,'w') as prod_write:
        prod_write.write('%start ' + str(start_symbol) + '\n')  # explicitly set the start marker
        for prod in productions:
            prod_write.write(str(prod)+"\n") #write productions to a file
    print("Production rules generated")



if __name__ == '__main__':

    if (len(sys.argv) >= 2):
        treebank_filename = sys.argv[1]
        pcfg_grammar_file = sys.argv[2]
    else:
        treebank_filename = '../data/parses.train'
        pcfg_grammar_file = '../data/pcfg_unk.out'
        print("Incorrect number of arguments")

    productions_file ="all_generated_productions.cfg"
    oov_productions_file = "oov_productions_file.cfg"
    treebank_records = open(treebank_filename, "r")
    #generate_all_productions(treebank_records,productions_file)
    grammar = nltk.data.load(productions_file,'cfg')
    get_terminal_frequency_and_count(grammar,0.2, oov_productions_file)
    oov_grammar = nltk.data.load(oov_productions_file, 'cfg')
    generate_pcfg(oov_grammar,pcfg_grammar_file)











