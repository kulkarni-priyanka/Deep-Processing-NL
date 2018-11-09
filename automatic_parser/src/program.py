'''
Read in an original context-free grammar.
Convert this grammar to Chomsky Normal Form.
Print out the rules of the converted grammar to a file.
'''

import nltk
import time

#declaring global variables
hybrid_pass =[]
unit_pass = []
long_pass =[]
counter = 1
dummy_list =[]

#function to create
def create_rule(lhs, rhs):
    for d_rule in dummy_list:
        if d_rule.rhs() == rhs:

            return d_rule
    dummy = nltk.grammar.Production(lhs,rhs)
    dummy_list.append(dummy)
    return dummy

def create_rule_with_RHS(rhs):
    global counter
    newKey = 'X' + str(counter)
    counter+=1
    lhs = nltk.Nonterminal(newKey)
    if isinstance(rhs, str):
        rhs = [rhs]

    return create_rule(lhs, rhs)

def get_prod_type(rhs):

    #Check if already conforms to CNF
    if (len(rhs) == 2 and nltk.grammar.is_nonterminal(rhs[0]) and nltk.grammar.is_nonterminal(rhs[1])):
        return 0

    if (len(rhs) == 1 and nltk.grammar.is_terminal(rhs[0])):
        return 0

    #Check if Unit Production
    if(len(rhs) == 1 and nltk.grammar.is_nonterminal(rhs[0])):
        return 2

    #Check if Hybrid Production
    for element in rhs:
        if nltk.grammar.is_terminal(element):
            return 1

    #Long Production
    return 3

def check_hybrid(prod):
    rhs = prod.rhs()

    rhs_NT = []

    for element in rhs:
        if nltk.grammar.is_terminal(element):
            print("Hybrid Rule")
            print(prod)
            new_rule = create_rule_with_RHS(element)
            hybrid_pass.append(new_rule)
            rhs_NT.append(new_rule.lhs())
        else:
            rhs_NT.append(element)
    hybrid_pass.append(create_rule(prod.lhs(),tuple(rhs_NT)))



def check_long(prod):
    rhs = prod.rhs()

    rhs_NT = []
    temp_rhs = list(rhs)
    new_rules =[]

    if len(rhs) > 2:
        while len(temp_rhs) !=2:

            new_rule =  create_rule_with_RHS(tuple([temp_rhs.pop(),temp_rhs.pop()]))
            new_rules.append(new_rule)
            temp_rhs = [new_rule.lhs()] + temp_rhs

        long_pass.append(create_rule(prod.lhs(),tuple(temp_rhs)))

        for rule in new_rules:
            long_pass.append(rule)

    else:
        long_pass.append(prod)


def get_sub_rules(prod, all_prod):
    nt = prod.rhs()
    rules_to_add =[]

    for rule in all_prod:
        if (rule.lhs() == nt[0]):
            if get_prod_type(rule.rhs()) == 0:
                rules_to_add.append(rule)
            elif len(rule.rhs()) == 1:
                sub_rules = get_sub_rules(rule, all_prod)
                rules_to_add = rules_to_add + list(sub_rules)
    return  rules_to_add


def check_unit(prod, all_prod):
    rhs = prod.rhs()
    if (len(rhs) == 1 and nltk.grammar.is_nonterminal(rhs[0])):
        rules_to_add = get_sub_rules(prod, all_prod)
        for rule in rules_to_add:
            unit_pass.append(create_rule(prod.lhs(),rule.rhs()))




if __name__ == "__main__":

    start_time = time.clock()
    input_cfg = nltk.load('../data/atis.cfg')

    for prod in input_cfg.productions():
        rhs = prod.rhs()
        if get_prod_type(rhs) == 0:
            hybrid_pass.append(prod)
        else:
            check_hybrid(prod)

    for prod in hybrid_pass:
        rhs = prod.rhs()
        if get_prod_type(rhs) == 0:
            long_pass.append(prod)
        else:
            check_long(prod)

    for prod in long_pass:
        rhs = prod.rhs()
        if get_prod_type(rhs) == 0:
            unit_pass.append(prod)
        else:
            check_unit(prod, long_pass)

    ng = nltk.grammar.CFG(input_cfg.start(), unit_pass)

    with open('atis_v2.cfg','w') as op_file:
        op_file.write('%start '+ str(input_cfg.start())+'\n')
        op_file.write("\n".join(str(x) for x in ng.productions()))

    print('Validation check using NLTK is_chomsky_normal_form() passed: '+ str(ng.is_chomsky_normal_form()))
    print('Total time taken in seconds: '+ str((time.clock() - start_time)))

















