import sys
import time
if __name__ == "__main__":

    if (len(sys.argv) >=2):
        input_grammar_filename = sys.argv[1]
        test_sentence_filename = sys.argv[2]
        output_filename = sys.argv[3]
    else:
        print("Incorrect number of arguments")

    start = time.clock()

    print('Parsing complete')
    print("Time taken :" + str(time.clock() - start))
