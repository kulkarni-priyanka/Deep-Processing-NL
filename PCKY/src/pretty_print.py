
from nltk.tree import Tree
import nltk

tr1 = str("(TOP (S_VP^<TOP> (S_VP_PRIME^<S_VP> (VB List) (NP^<S_VP_PRIME> (NP_PRIME^<NP> (NP^<NP_PRIME> (DT the) (NNS flights)) (PP^<NP_PRIME> (IN from) (NP_NNP Baltimore))) (PP^<NP> (TO to) (NP_NNP Seattle)))) (NP^<S_VP> (NP_PRIME^<NP> (DT that) (NN stop)) (PP^<NP> (IN in) (NP_NNP Minneapolis)))) (PUNC .))")
s1 = Tree.fromstring(tr1)
s1.pretty_print()
'''
text = nltk.word_tokenize("Arriving")
x = nltk.pos_tag(text)
print(x)
text = nltk.word_tokenize("Arriving")
y = nltk.pos_tag(text)[0]
print(nltk.grammar.Nonterminal(str(y[1])))
'''