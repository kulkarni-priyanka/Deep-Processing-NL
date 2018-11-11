import gensim
import nltk
from gensim.models import Word2Vec
from scipy.stats.stats import spearmanr
import sys
import time
from nltk.corpus import stopwords
import string

if __name__ == "__main__":

    if (len(sys.argv) >=2):
        cbow_window = sys.argv[1]
        judgment_filename = sys.argv[2]
        output_filename = sys.argv[3]

    else:
        cbow_window = 2
        judgment_filename = "../data/mc_similarity.txt"
        output_filename = "../data/ hw7_sim_2_CBOW_output_v3.txt "
        print("Incorrect number of arguments")

    start = time.clock()

    cbow_window = int(cbow_window)


    sentences = nltk.corpus.brown.sents()

    final_sents = []

    for sent in sentences:
        #convert to lower
        sent = [w.lower() for w in sent if w.lower() not in string.punctuation]
        final_sents.append(sent)

    #create word2vec model
    model = Word2Vec(final_sents, window = cbow_window, min_count=1, workers=1,iter=5)

    w2v_embeddings = model
    human_scores = []
    embedding_scores = []
    with open(output_filename,'w') as op_file:
        with open(judgment_filename, 'r') as hj_file:
            judgments = hj_file.readlines()
            for line in judgments:
                line = line.split(sep=',')
                word_1 = line[0]
                word_2 = line[1]
                human_scores.append(float(line[2]))
                #compute similarity
                e_score = w2v_embeddings.wv.similarity(word_1,word_2)
                embedding_scores.append(e_score)
                op_file.write(word_1+","+word_2+":"+str(e_score)+"\n")

            #compute correlation
            op_file.write("correlation:"+ str(spearmanr(human_scores, embedding_scores).correlation)+"\n")

    print("Time taken :"+ str(time.clock()-start))