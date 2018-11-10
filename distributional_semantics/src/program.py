import nltk
import sys
from collections import defaultdict
import string
from nltk.corpus import stopwords
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats.stats import spearmanr

class CollocationMatrix(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._word_mapping = {}  # Where we'll store string->int mapping.

    def word_id(self, word, store_new=False):
        """
        Return the integer ID for the given vocab item. If we haven't
        seen this vocab item before, give ia a new ID. We can do this just
        as 1, 2, 3... based on how many words we've seen before.
        """
        if word not in self._word_mapping:
            if store_new:
                self._word_mapping[word] = len(self._word_mapping)
                self[self._word_mapping[word]] = defaultdict(int)  # Also add a new row for this new word.
            else:
                return None
        return self._word_mapping[word]

    def add_pair(self, w_1, w_2):
        """
        Add a pair of colocated words into the coocurrence matrix.
        """
        w_id_1 = self.word_id(w_1, store_new=True)
        w_id_2 = self.word_id(w_2, store_new=True)
        self[w_id_1][w_id_2] += 1  # Increment the count for this collocation

    def get_pair(self, w_1, w_2):
        """
        Return the colocation for w_1, w_2
        """
        w_1_id = self.word_id(w_1)
        w_2_id = self.word_id(w_2)
        if w_1_id and w_2_id:
            return self[w_1_id][w_2_id]
        else:
            return 0

    def get_row(self, word):
        word_id = self.word_id(word)
        if word_id is not None:
            return self.get(word_id)
        else:
            return defaultdict(int)

    def get_row_sum(self, word):
        """
        Get the number of total contexts available for a given word
        """
        return sum(self.get_row(word).values())

    def get_col_sum(self, word):
        """
        Get the number of total contexts a given word occurs in
        """
        f_id = self.word_id(word)
        return sum([self[w][f_id] for w in self.keys()])

    @property
    def total_sum(self):
        return sum([self.get_row_sum(w) for w in self._word_mapping.keys()])


def calculate_ppmi(w, f):
    sum_all_context = matrix.total_sum
    word_count = matrix.get_row_sum(w)
    context_count = matrix.get_col_sum(f)
    joint_count = matrix.get_pair(w, f)

    if sum_all_context:
        p_w = word_count / float(sum_all_context)
        p_f = context_count / float(sum_all_context)
        p_w_f = joint_count / float(sum_all_context)

    if p_w * p_f > 0:
        ratio = (p_w_f) / (p_w * p_f)
        if ratio > 0:
            return max(math.log2(ratio), 0)
        else:
            return 0

    else:
        return 0

def calculate_freq(w, f):
    return matrix.get_pair(w,f)

if __name__ == "__main__":

    if (len(sys.argv) >=2):
        judgement_filename = sys.argv[1]
        sentences_filename = sys.argv[2]

    else:

        judgement_filename = "../data/mc_similarity.txt"
        output_filename = "../data/hw7_output.txt"
        print("Incorrect number of arguments")

    window_size = 3
    sent_limit = 10000
    matrix = CollocationMatrix()
    stopwords = nltk.corpus.stopwords.words('english')

    brown_sents = nltk.corpus.brown.sents()

    for sent in brown_sents[:sent_limit]:
        sent = [w for w in sent if w.lower() not in stopwords]
        sent = [w for w in sent if w.lower() not in string.punctuation]
        for i, word in enumerate(sent):
            # Increment the count of words we've seen.
            for j in range(-window_size, window_size + 1):
                # Skip counting the word itself.
                if j == 0:
                    continue

                # At the beginning and end of the sentence,
                # you can either skip counting, or add a
                # unique "<START>" or "<END>" token to indicate
                # the word being colocated at the beginning or
                # end of sentences.
                if len(sent) > i + j > 0:
                    word_1 = sent[i].lower()
                    word_2 = sent[i + j].lower()

                    matrix.add_pair(word_1, word_2)

        vocab_size = len(matrix._word_mapping.keys())

    inv_word_map = {v: k for k, v in matrix._word_mapping.items()}

    print(vocab_size)
    ppmi_matrix = np.zeros((vocab_size, vocab_size ))

    vocab = list(matrix._word_mapping.keys())

    judgement_vocab = []
    human_scores = []

    with open(judgement_filename, 'r') as hj_file:
        judgements = hj_file.readlines()
        for line in judgements:
            line = line.split(sep=',')
            judgement_vocab.append(line[0])
            judgement_vocab.append(line[1])
            human_scores.append(float(line[2]))

    '''
    for word_1 in judgement_vocab:
        for word_2 in vocab:
            if matrix.get_pair(word_1,word_2) > 0:
                ppmi = calculate_ppmi(word_1, word_2)
                w_id_1 = matrix.word_id(word_1)
                w_id_2 = matrix.word_id(word_2)
                ppmi_matrix[w_id_1][w_id_2] = ppmi


    np.save("ppmi_10K.npy",ppmi_matrix)
    '''


    ppmi_matrix = np.load("ppmi_10K.npy")

    cos_sim_scores = []

    with open(output_filename,'w') as op_write:
        with open(judgement_filename,'r') as hj_file:
            judgements = hj_file.readlines()
            for line in judgements:
                line = line.split(sep=',')

                word_1 = line[0]
                w_id_1 = matrix.word_id(word_1)
                write_feature = ""
                write_feature += word_1 + " "
                if w_id_1:
                    arr_w1 = ppmi_matrix[w_id_1]
                    a1 = arr_w1.argsort()[-10:][::-1]
                    for index in a1:
                        write_feature += str(inv_word_map[index])+":"+str(ppmi_matrix[w_id_1][index])+" "
                    op_write.write(write_feature+"\n")
                else:
                    a1 = np.zeros(10)
                    op_write.write(write_feature + "\n")


                word_2 = line[1]
                w_id_2 = matrix.word_id(word_2)
                if w_id_2:
                    arr_w2 = ppmi_matrix[w_id_2]
                    a2 = arr_w2.argsort()[-10:][::-1]
                    write_feature = ""
                    write_feature += word_2 + " "
                    for index in a2:
                        write_feature += str(inv_word_map[index]) + ":" + str(ppmi_matrix[w_id_2][index]) + " "
                    op_write.write(write_feature + "\n")
                else:
                    a2 = np.zeros(10)
                    op_write.write(write_feature + "\n")

                if w_id_1 and w_id_2:
                    cos_sim = cosine_similarity([ppmi_matrix[w_id_1]],[ppmi_matrix[w_id_2]])
                    cos_sim_scores.append(cos_sim)
                    op_write.write(word_1+","+word_2+":"+str(cos_sim)+"\n")
                else:
                    op_write.write("Out of Vocabulary")




    print('Parsing complete')
