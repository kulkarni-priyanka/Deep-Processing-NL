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
import operator

wnic = None


def get_reznik_similarity(wordSynset, contextSynset):
    #print(wordSynset._pos)
    if wordSynset._pos != 'n':
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

    #subsumer_ic = max(information_content(s, ic) for s in subsumers)

    for subsumer in subsumers:
        if subsumer._pos != 'n':
            continue
        result = nltk.corpus.reader.information_content(subsumer, wnic)
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

    start = time.clock()

    if information_content_file_type =="nltk":
        wnic = wordnet_ic.ic('ic-brown-resnik-add1.dat')



    with open(output_filename,'w') as op_file:

        with open(wsd_test_filename,'r') as wsd_file:
            for line in wsd_file:
                line = line.strip('\n')
                line = line.split('\t')
                probe_word = line[0]

                noun_groups = line[1].split(',')

                wordSynsets = wn.synsets(probe_word)
                max ={}

                for noun_group in noun_groups:
                    max_value = 0
                    max_probe_synset = None
                    max_context_synset = None
                    max_ic_subsumer = None

                    ngSynsets = wn.synsets(noun_group)

                    for ws in wordSynsets:
                        for ns in ngSynsets:
                            (value,sub) = get_reznik_similarity(ws,ns)

                            if value > max_value:
                                max_value = value
                                max_probe_synset = ws
                                max_context_synset = ns
                                max_ic_subsumer = sub

                    # set the score

                    write_line = "("+probe_word+","+noun_group+","+str(max_value)+") "
                    op_file.write(write_line)

                    if max_probe_synset in max:
                        max[max_probe_synset] = max[max_probe_synset] + max_value
                    else:
                        max[max_probe_synset] = max_value

                sorted_max = sorted(max.items(), key=operator.itemgetter(1), reverse=True)
                op_file.write("\n"+str(sorted_max[0][0]._name)+"\n")
                


        human_scores =[]
        resnik_scores =[]

        with open(judgment_file,'r') as judgments:
            for line in judgments:
                line = line.strip('\n')
                line = line.split(',')
                probe_word = line[0]
                context = line[1]
                human_score = float(line[2])
                human_scores.append(human_score)

                wordSynsets = wn.synsets(probe_word)
                ngSynsets = wn.synsets(context)

                max ={}
                max_value = 0
                max_probe_synset = None
                max_context_synset = None
                max_ic_subsumer = None

                for ws in wordSynsets:
                    for ns in ngSynsets:
                        (value, sub) = get_reznik_similarity(ws, ns)

                        if value > max_value:
                            max_value = value
                            max_probe_synset = ws
                            max_context_synset = ns
                            max_ic_subsumer = sub

                # set the score
                '''
                if max_probe_synset in max:
                    max[max_probe_synset] = max[max_probe_synset] + max_value
                else:
                    max[max_probe_synset] = max_value

                sorted_max = sorted(max.items(), key=operator.itemgetter(1), reverse=True)
                '''

                similarity_score = max_value
                resnik_scores.append(similarity_score)

                write_line = "" + probe_word + "," + context + ":" + str(similarity_score) + "\n"
                op_file.write(write_line)
            op_file.write("Correlation:" + str(spearmanr(human_scores, resnik_scores).correlation) + "\n")





    print('Parsing complete')
    print("Time taken :" + str(time.clock() - start))
