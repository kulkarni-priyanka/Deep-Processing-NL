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
    '''
    creating ic file for only words in the wsd_contexts.txt and the mc_similarity.txt files.
    :param wsd_test_filename: file wsd_contexts.txt
    :param judgment_file: mc_similarity.txt
    :param icfile: name of the output ic file
    :return: ic dictionary which can be read by nltk.corpus.reader.information_content
    '''

    brown_corpus = brown.tagged_sents() #extract tagged sentences which are of the form (word,tag). We are doing this so that we can pick words with only the NN tags
    senses_list = []
    all_words = {}

    #since we are creating ic file for only words in the two given files, extract all words, contexts and their subsumers and store it in a list

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

    #extract the lemma from all senses
    for sense in senses_list:
        lemma, pos, index = sense._name.split('.')
        all_words[lemma] = None

    #add1 smoothing
    for aw in all_words:
        all_words[aw] = 1

    noun_tags = ['NN']

    #increment counts for words that occur in the brown corpus
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

            #compute Ic as per the formula in the class notes
            logProb = -math.log(float(count) / total)

            for synset in synsets:
                line = str(synset._offset) + 'n' + ' ' + str(logProb) + '\n'
                newIcFile.write(line)

    ic = create_ic_dictionary(icfile)
    return ic


def create_ic_dictionary(icfile):
    #code reused from: http://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html
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

    #when no subsumers are found return 0 as max_value and None as max_subsumer
    if len(subsumers) == 0:
        return (max_value, max_subsumer)

    #subsumer_ic = max(information_content(s, ic) for s in subsumers)


    for subsumer in subsumers:
        if subsumer._pos != 'n': #choose only subsumers with pos as noun
            continue
        result = nltk.corpus.reader.information_content(subsumer, wnic) #get the ic for the subsumer

        #if ic of the current subsumer is greater than the max value, set the new value as max
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
        print("Incorrect number of arguments")

    start = time.clock()

    #select the right ic file based on the input parameters
    if information_content_file_type =="nltk":
        wnic = wordnet_ic.ic('ic-brown-resnik-add1.dat')
    else:
        wnic = create_ic(wsd_test_filename,judgment_file,'hw8_myic.txt')

    with open(output_filename,'w') as op_file:

        #creating a list to store the answers obtained by the algorithm
        wsd_answers_obtained = []

        with open(wsd_test_filename,'r') as wsd_file:

            for line in wsd_file:
                line = line.strip('\n')
                line = line.split('\t')
                probe_word = line[0] #extract probe word
                noun_groups = line[1].split(',') #estract noun groups

                wordSynsets = wn.synsets(probe_word) #get all synsets of the probe_word
                max ={} #stores the values in preference of a word sense

                for noun_group in noun_groups:
                    max_value = 0
                    max_probe_synset = None
                    max_context_synset = None
                    max_ic_subsumer = None

                    ngSynsets = wn.synsets(noun_group) #get all synsets of the noun_group

                    for ws in wordSynsets: #for all values of probe_word synsets
                        for ns in ngSynsets: #for all values of context word synsets
                            (value,sub) = get_reznik_similarity(ws,ns) #get resnik similarity value and the subsumer that gives that value

                            #if value greater than max_value, set the new values as max for the given word, noun_group tuple
                            if value > max_value:
                                max_value = value
                                max_probe_synset = ws
                                max_context_synset = ns
                                max_ic_subsumer = sub

                    #print the similarity score
                    write_line = "("+probe_word+", "+noun_group+", "+str(max_value)+") "
                    op_file.write(write_line)

                    #increment the value of the probe word sense which has the max value
                    if max_probe_synset in max:
                        max[max_probe_synset] += max_value
                    else:
                        max[max_probe_synset] = max_value

                #sort descencing the max dictionary to obtain the word sense that is most preferred given the context words
                sorted_max = sorted(max.items(), key=operator.itemgetter(1), reverse=True)

                final_sense = sorted_max[0][0]._name #print the preferred sense

                #extract and print the definitions for the sense preferred
                definition = sorted_max[0][0]._definition
                print(str(final_sense) + "\t" + str(definition))

                wsd_answers_obtained.append(final_sense)
                op_file.write("\n"+str(final_sense)+"\n")

        with open('wsd_contexts.txt.gold','r') as wsd_gold:
            wsd_answers_gold = []
            for line in wsd_gold:
                wsd_answers_gold.append(line.split()[0])
        total_records = len(wsd_answers_gold)

        list_common = []
        #compare the results obtained by the algorith with the gold file
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
                probe_word = line[0] #extract probe word
                context = line[1] #extract contexts
                human_score = float(line[2]) #get human score
                human_scores.append(human_score)

                wordSynsets = wn.synsets(probe_word) #get all synsets for the probe_word
                ngSynsets = wn.synsets(context) #get all synsets for the context word

                max ={}
                max_value = 0
                max_probe_synset = None
                max_context_synset = None
                max_ic_subsumer = None

                for ws in wordSynsets:
                    for ns in ngSynsets:
                        (value, sub) = get_reznik_similarity(ws, ns) #get resnik similarity for ws and ns

                        # if value greater than max_value, set the new values as max for the given word, noun_group tuple
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

            #compute the correlation using the scipy.stats spearman
            op_file.write("Correlation:" + str(spearmanr(human_scores, resnik_scores).correlation) + "\n")

    print('Parsing complete')
    print("Time taken :" + str(time.clock() - start))
