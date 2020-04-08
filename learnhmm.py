'''

learnhmm.py

Implementation of hidden Markov model.

Learn the parameters needed for the forward backward algorithm.

'''

import sys
import math
import numpy as np

Debug = True

# Normalize by rows
def norm_rows(W):
    return W / W.sum(axis=1)[:, np.newaxis]

class hmm_train(object):
    def __init__(self, index2word, index2tag, prior_out, emit_out, trans_out):
        self.index2word = index2word
        self.index2tag = index2tag
        self.prior_out = prior_out
        self.emit_out = emit_out
        self.trans_out = trans_out
        self.word2idx = dict()
        self.tag2idx = dict()

    def train_model(self, train_input):
        # load word2idx
        with open(self.index2word, 'r') as f_idx2word:
            i = 0
            for line in f_idx2word:
                self.word2idx[line] = i
                i += 1

        # load tag2idx
        with open(self.index2tag, 'r') as f_index2tag:
            i = 0
            for line in f_index2tag:
                self.tag2idx[line] = i
                i += 1

        # parameter initialization
        self.N = len(self.tag2idx)
        self.M = len(self.word2idx)

        self.pi = np.ones(self.N)
        self.A = np.ones((self.N, self.N))
        self.B = np.ones((self.N, self.M))

        # read from the training data
        with open(train_input, 'r') as f:
            for line in f:
                sentence = []
                words = line.strip().split(' ')
                i = 0
                for ele in words:
                    word_idx = self.word2idx[ele.split('_')[0]]
                    tag_idx = self.word2idx[ele.split('_')[1]]

                    # update intermediate counts
                    if i == 0:
                        self.pi[tag_idx] += 1
                    else:
                        self.A[sentence[i - 1][1]][tag_idx] += 1

                    self.B[tag_idx][word_idx]

                    sentence.append([word_idx, tag_idx])
                    i += 1

        # normalize parameters
        self.pi = norm_rows(self.pi)
        self.A = norm_rows(self.A)
        self.B = norm_rows(self.B)

        if Debug:
            print("pi: ", self.pi)
            print("A: ", self.A)
            print("B: ", self.B)

        # write the parameters to files
        with open(prior_out, 'w') as f_prior:
            for j in range(self.N):
                f_prior.write(str(self.pi[j]) + "\n")
        
        with open(trans_out, 'w') as f_trans:
            for j in range(self.N):
                for k in range(self.N):
                    f_trans.write(str(self.A[j][k]) + " ")
                f_trans.write("\n")
            
        with open(emit_out, 'w') as f_emit:
            for j in range(self.N):
                for k in range(self.N):
                    f_emit.write(str(self.B[j][k]) + " ")
                f_emit.write("\n")


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("The number of command parameters is incorrect.")
        exit(-1)

    train_input = sys.argv[1]  # path to the training input .csv file
    index2word = sys.argv[2] # path to the .txt that specifies the dictionary mapping from words to indices. 
                             # The tags are ordered by index, with the first word having index of 1, the second word having index of 2, etc.
    index2tag = sys.argv[3] # path to the .txt that specifies the dictionary mapping from tags to indices. 
                            # The tags are ordered by index, with the first tag having index of 1, the second tag having index of 2, etc.
    prior_out = sys.argv[4] # path to output .txt file to which the estimated prior (pi) will be written. (same format as hmmprior.txt)
    emit_out = sys.argv[5] # path to output .txt file to which the emission probabilities (B) will be written. (same format as hmmemit.txt)
    trans_out = sys.argv[6] # path to output .txt file to which the transition probabilities (A) will be written. (same format as hmmtrans.txt)

    # build and init 
    model = hmm_train(index2word, index2tag, prior_out, emit_out, trans_out)

    # train and and write model parameter to output files
    model.train_model(train_input)

