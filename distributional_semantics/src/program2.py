import gensim
import nltk
from gensim.models import Word2Vec
from scipy.stats.stats import spearmanr
import sys
import time

if __name__ == "__main__":

    if (len(sys.argv) >=2):
        cbow_window = sys.argv[1]
        judgment_filename = sys.argv[2]
        output_filename = sys.argv[3]

    else:
        cbow_window = 2
        judgment_filename = "../data/mc_similarity.txt"
        output_filename = "../data/ hw7_sim_2_CBOW_output.txt "
        print("Incorrect number of arguments")

    start = time.clock()

    cbow_window = int(cbow_window)

    sentences = nltk.corpus.brown.sents()

    #test with different corpus
    #convert tow lercase and remove punctuation
    #check if gensim similarity is log2

    model = Word2Vec(sentences, window = cbow_window, min_count=1, workers=1)
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
                e_score = w2v_embeddings.wv.similarity(word_1,word_2)
                embedding_scores.append(e_score)
                op_file.write(word_1+","+word_2+","+str(e_score)+"\n")

            op_file.write("correlation:"+ str(spearmanr(human_scores, embedding_scores).correlation)+"\n")

    print("Time taken :"+ str(time.clock()-start))



