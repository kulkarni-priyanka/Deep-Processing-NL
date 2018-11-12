import nltk
import sys
from collections import defaultdict
import string
from nltk.corpus import stopwords
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats.stats import spearmanr
import time


row_sum_dict ={}
col_sum_dict ={}

#creating the collocation matrix to store frequencies
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


#caching sums
def cache_sums(matrix, vocab):
    for word in vocab:
        row_sum_dict[word] = matrix.get_row_sum(word)
        col_sum_dict[word] = matrix.get_col_sum(word)

#PPMI compuation
def calculate_ppmi(w, f):
    sum_all_context = matrix.total_sum
    word_count = row_sum_dict[w]
    context_count = col_sum_dict[f]
    joint_count = matrix.get_pair(w, f)

    if sum_all_context:
        p_w = word_count / float(sum_all_context)
        p_f = context_count / float(sum_all_context)
        p_w_f = joint_count / float(sum_all_context)

    if p_w * p_f > 0:
        ratio = (p_w_f) / (p_w * p_f)
        if ratio > 0:
            return max(math.log2(ratio), 0) #return only positive values
        else:
            return 0

    else:
        return 0

def calculate_freq(w, f):
    return matrix.get_pair(w,f)

if __name__ == "__main__":

    if (len(sys.argv) >=2):
        window = sys.argv[1]
        weighting = sys.argv[2]
        judgement_filename = sys.argv[3]
        output_filename = sys.argv[4]

    else:
        print("Incorrect number of arguments")

    start = time.clock()

    if weighting in ('FREQ', 'PMI'):
        window_size = int(window)

        matrix = CollocationMatrix()
        stopwords = nltk.corpus.stopwords.words('english')

        brown_sents = nltk.corpus.brown.sents()

        print(len(brown_sents))

        for sent in brown_sents:
            sent = [w for w in sent if w.lower() not in stopwords] #remove stopwords
            sent = [w.lower().strip(string.punctuation) for w in sent] #convert to lower
            sent = [w for w in sent if w != ""] #remove blanks
            for i, word in enumerate(sent):
                # Increment the count of words we've seen.
                for j in range(-window_size, window_size + 1):
                    # Skip counting the word itself.
                    if j == 0:
                        continue

                    if len(sent) > i + j > 0:
                        word_1 = sent[i].lower()
                        word_2 = sent[i + j].lower()
                        matrix.add_pair(word_1, word_2)

        #compute vocab size
        vocab_size = len(matrix._word_mapping.keys())
        inv_word_map = {v: k for k, v in matrix._word_mapping.items()}
        print(vocab_size)

        #initialize weight matrix
        weighted_matrix = np.zeros((vocab_size, vocab_size ))
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

        if weighting =='PMI':
            #populate the ppmi weights for each word in judgement file
            for word_1 in judgement_vocab:
                if word_1 not in row_sum_dict:
                    row_sum_dict[word_1] = matrix.get_row_sum(word_1)
                    col_sum_dict[word_1] = matrix.get_col_sum(word_1)
                for word_2 in vocab:
                    if matrix.get_pair(word_1,word_2) > 0:
                        if word_2 not in col_sum_dict:
                            col_sum_dict[word_2] = matrix.get_col_sum(word_2)
                        ppmi = calculate_ppmi(word_1, word_2)
                        w_id_1 = matrix.word_id(word_1)
                        w_id_2 = matrix.word_id(word_2)
                        weighted_matrix[w_id_1][w_id_2] = ppmi

        else:
            for word_1 in judgement_vocab:

                for word_2 in vocab:
                    freq = matrix.get_pair(word_1, word_2)
                    if freq > 0:
                        w_id_1 = matrix.word_id(word_1)
                        w_id_2 = matrix.word_id(word_2)
                        weighted_matrix[w_id_1][w_id_2] = freq


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
                        arr_w1 = weighted_matrix[w_id_1]
                        a1 = arr_w1.argsort()[-10:][::-1]
                        for index in a1:
                            write_feature += str(inv_word_map[index])+":"+str(weighted_matrix[w_id_1][index])+" "
                        op_write.write(write_feature+"\n")
                    else:
                        a1 = np.zeros(10)
                        op_write.write(write_feature + "\n")


                    word_2 = line[1]
                    w_id_2 = matrix.word_id(word_2)
                    write_feature = ""
                    write_feature += word_2 + " "
                    if w_id_2:
                        arr_w2 = weighted_matrix[w_id_2]
                        a2 = arr_w2.argsort()[-10:][::-1]
                        for index in a2:
                            write_feature += str(inv_word_map[index]) + ":" + str(weighted_matrix[w_id_2][index]) + " "
                        op_write.write(write_feature + "\n")
                    else:
                        a2 = np.zeros(10)
                        op_write.write(write_feature + "\n")

                    if w_id_1 and w_id_2:
                        #compute similarity
                        cos_sim = cosine_similarity([weighted_matrix[w_id_1]],[weighted_matrix[w_id_2]])
                        cos_sim_scores.append(cos_sim[0][0])
                        op_write.write(word_1+","+word_2+":"+str(cos_sim[0][0])+"\n")
                    else:

                        cos_sim_scores.append(0)
                        op_write.write(word_1 + "," + word_2 + ":" + str(0) + "\n")

            print(len(human_scores))
            print(len(cos_sim_scores))

            op_write.write("correlation:" + str(spearmanr(human_scores, cos_sim_scores).correlation) + "\n")
    else:
        print("Incorrect weighting option")

    print('Parsing complete')
    print("Time taken :" + str(time.clock() - start))
