import nltk
import sys
import string
from nltk.corpus import wordnet_ic
from nltk.corpus import wordnet as wn
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats.stats import spearmanr
import time


def reznik(self, wordSynset, contextSynset):
    if wordSynset.pos != noun:
        return (0, None)

    # get the hypernyms

    subsumers = wordSynset.common_hypernyms(contextSynset)

    maxValue = 0

    maxSubsumer = None

    # handle when 0

    if len(subsumers) == 0:
        return (maxValue, maxSubsumer)

    maxValue = 0

    maxSubsumer = None

    for subsumer in subsumers:

        if subsumer.pos != noun:
            continue

        result = nltk.corpus.reader.information_content(subsumer, self.ic)

        if result > maxValue:
            maxValue = result

            maxSubsumer = subsumer

    return (maxValue, maxSubsumer)




if __name__ == "__main__":

    if (len(sys.argv) >=2):
        information_content_file_type = sys.argv[1]
        wsd_test_filename = sys.argv[2]
        judgment_file = sys.argv[3]
        output_filename = sys.argv[4]

    else:
        information_content_file_type = "nltk"
        wsd_test_filename = "../data/wsd_contexts.txt"
        judgment_file = "../data/mc_similarity.txt"
        output_filename = "../data/hw8_output.txt"
        #print("Incorrect number of arguments")

    if information_content_file_type =="nltk":
        wnic = wordnet_ic.ic('ic-brown-resnik-add1.dat')




    with open(wsd_test_filename,'r') as wsd_file:
        for line in wsd_file:
            line = line.split('\t')
            probe_word = line[0]
            noun_groups = line[1].split(',')
            wordSynsets = wn.synsets(probe_word)

            for noun_group in noun_groups:
                break





    start = time.clock()



    print('Parsing complete')
    print("Time taken :" + str(time.clock() - start))
