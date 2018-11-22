import nltk
import sys
import string
from nltk.corpus import wordnet_ic
from nltk.corpus import wordnet as wn
import math
from scipy.stats.stats import spearmanr
import time
import operator
from collections import defaultdict
from nltk.corpus import brown

wnic = None

def create_ic(wsd_test_filename,judgment_file,icfile):
    brown_corpus = brown.tagged_sents()
    senses_list = []
    all_words = {}
    with open(wsd_test_filename, 'r') as wsd_file:
        for line in wsd_file:
            line = line.strip('\n')
            line = line.split('\t')
            probe_word = line[0]
            noun_groups = line[1].split(',')

            for ws in wn.synsets(probe_word, pos=wn.NOUN):
                senses_list.append(ws)
                for context in noun_groups:
                    for cs in wn.synsets(context, pos=wn.NOUN):
                        senses_list.append(cs)
                        subsumers = ws.common_hypernyms(cs)
                        for subsumer in subsumers:
                            if subsumer._pos == 'n':
                                senses_list.append(subsumer)

    with open(judgment_file, 'r') as judge_file:
        for line in judge_file:
            line = line.strip('\n')
            line = line.split(',')
            probe_word = line[0]
            context = line[1]

            for ws in wn.synsets(probe_word, pos=wn.NOUN):
                senses_list.append(ws)
                for cs in wn.synsets(context, pos=wn.NOUN):
                    senses_list.append(cs)
                    subsumers = ws.common_hypernyms(cs)
                    for subsumer in subsumers:
                        if subsumer._pos == 'n':
                            senses_list.append(subsumer)

    for sense in senses_list:
        lemma, pos, index = sense._name.split('.')
        all_words[lemma] = None

    for aw in all_words:
        all_words[aw] = 1

    noun_tags = ['NN']

    for sent in brown_corpus:
        for w, t in sent:
            if w in all_words and t in noun_tags:
                all_words[w] += 1

    total_words = sum(all_words.values())

    with open(icfile, 'w') as newIcFile:

        #took help from this stackoverflow post : https://stackoverflow.com/questions/32436663/create-information-content-corpora-to-be-used-by-webnet-from-a-custom-dump

        # Hash code of WordNet 3.0
        newIcFile.write('wnver::eOS9lXC6GvMWznF1wkZofDdtbBU' + '\n')
        newIcFile.write('1740n ' + str(total_words) + ' ROOT\n')

        for word in all_words:
            synsets = wn.synsets(word)
            count = all_words[word]
            total = total_words

            logProb = -math.log(float(count) / total)

            for synset in synsets:
                line = str(synset._offset) + 'n' + ' ' + str(logProb) + '\n'
                newIcFile.write(line)

    ic = create_ic_dictionary(icfile)

    return ic


def create_ic_dictionary(icfile):
    #http://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html
    """
    Load an information content file from the wordnet_ic corpus
    and return a dictionary.  This dictionary has just two keys,
    NOUN and VERB, whose values are dictionaries that map from
    synsets to information content values.

    :type icfile: str
    :param icfile: The name of the wordnet_ic file (e.g. "ic-brown.dat")
    :return: An information content dictionary
    """
    ic = {}
    ic['n'] = defaultdict(float)
    ic['v'] = defaultdict(float)
    for num, line in enumerate(open(icfile)):
        if num == 0:  # skip the header
            continue
        fields = line.split()
        offset = int(fields[0][:-1])
        value = float(fields[1])
        pos = 'n'
        if len(fields) == 3 and fields[2] == "ROOT":
            # Store root count.
            ic[pos][0] += value
        if value != 0:
            ic[pos][offset] = value
    return ic


def get_reznik_similarity(ws, cs):
    '''
    :param ws: word synset
    :param cs: context synset
    :return: subsumer with the max information content, information content value
    '''

    #print(wordSynset._pos)

    # get the common hypernyms
    subsumers = ws.common_hypernyms(cs)
    max_value = 0
    max_subsumer = None

    # handle when 0

    if len(subsumers) == 0:
        return (max_value, max_subsumer)

    #subsumer_ic = max(information_content(s, ic) for s in subsumers)

    for subsumer in subsumers:
        if subsumer._pos != 'n':
            continue
        result = nltk.corpus.reader.information_content(subsumer, wnic)
        if result > max_value:
            max_value = result
            max_subsumer = subsumer

    return (max_value, max_subsumer)


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
        output_filename = "../data/hw8_output_nltk2.txt"
        #print("Incorrect number of arguments")

    start = time.clock()

    if information_content_file_type =="nltk":
        wnic = wordnet_ic.ic('ic-brown-resnik-add1.dat')
    else:
        wnic = create_ic(wsd_test_filename,judgment_file,'hw8_myic_ptb.txt')


    with open(output_filename,'w') as op_file:

        wsd_answers_obtained = []

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
                        max[max_probe_synset] += max_value
                    else:
                        max[max_probe_synset] = max_value

                sorted_max = sorted(max.items(), key=operator.itemgetter(1), reverse=True)

                fe = sorted_max[0][0]._definition
                final_sense = sorted_max[0][0]._name
                print(str(final_sense)+"\t"+str(fe))
                wsd_answers_obtained.append(final_sense)
                op_file.write("\n"+str(final_sense)+"\n")

        with open('../data/wsd_contexts.txt.gold','r') as wsd_gold:
            wsd_answers_gold = []
            for line in wsd_gold:
                wsd_answers_gold.append(line.split()[0])
        total_records = len(wsd_answers_gold)

        list_common = []
        for a, b in zip(wsd_answers_obtained, wsd_answers_gold):
            if a == b:
                list_common.append(a)
        correct_obtained = len(list_common)
        print("Accuracy is: "+ str((correct_obtained*100)/float(total_records)))


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
                similarity_score = max_value
                resnik_scores.append(similarity_score)

                write_line = "" + probe_word + "," + context + ":" + str(similarity_score) + "\n"
                op_file.write(write_line)
            op_file.write("Correlation:" + str(spearmanr(human_scores, resnik_scores).correlation) + "\n")

    print('Parsing complete')
    print("Time taken :" + str(time.clock() - start))
