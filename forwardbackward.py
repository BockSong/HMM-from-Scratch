'''

forwardbackward.py

Implementation of hidden Markov model.

Run the forward backward algorithm.

'''

import sys
import math
import numpy as np

Debug = True
PrintGrad = True

class hmm_eval(object):
    def __init__(self, index2word, index2tag, hmmprior, hmmemit, hmmtrans, metrics_out):
        self.index2word = index2word
        self.index2tag = index2tag
        self.hmmprior = hmmprior
        self.hmmemit = hmmemit
        self.hmmtrans = hmmtrans
        self.metrics_out = metrics_out
        self.word2idx = dict()
        self.tag2idx = dict()

    # open file and read parameters
    def build_model(self):
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

        # read the parameters
        with open(hmmprior, 'w') as f_prior:
            for j in range(self.N):
                num = f_prior.readline().strip()
                self.pi[j] = float(num)
        
        with open(hmmemit, 'w') as f_trans:
            for j in range(self.N):
                words = f_trans.readline().strip().split(' ')
                for k in range(self.N):
                    self.A[j][k] = float(words[k])
            
        with open(hmmtrans, 'w') as f_emit:
            for j in range(self.N):
                words = f_emit.readline().strip().split(' ')
                for k in range(self.N):
                    self.B[j][k] = float(words[k])



    # predict y given an array x
    # not used
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        sm = np.exp(x) / np.sum(np.exp(x), axis=0)
        return np.argmax(sm)

    def evaluate(self, in_path, out_path, write = False):
        total_loss, error, total = 0., 0., 0.

        with open(in_path, 'r') as f_in:
            with open(out_path, 'a') as f_out:
                for line in f_in:
                    split_line = line.strip().split(',')
                    y = int(split_line[0])
                    x = np.asarray(split_line[1:], dtype=int)

                    for layer in self.layers:
                        x = layer.forward(x)
                    loss, pred = self.criterion.forward(x, y)

                    total_loss += loss
                    if pred != y:
                        error += 1
                    if write:
                        f_out.write(str(pred) + "\n")
                    total += 1

        return total_loss / total, error / total


if __name__ == '__main__':
    if len(sys.argv) != 8:
        print("The number of command parameters is incorrect.")
        exit(-1)

    test_input = sys.argv[1] # path to the test input .txt file that will be evaluated by your forward backward algorithm
    index2word = sys.argv[2] # path to the .txt that specifies the dictionary mapping from words to indices. 
                             # The tags are ordered by index, with the first word having index of 1, the second word having index of 2, etc.
    index2tag = sys.argv[3] # path to the .txt that specifies the dictionary mapping from tags to indices. 
                            # The tags are ordered by index, with the first tag having index of 1, the second tag having index of 2, etc.
    hmmprior = sys.argv[4] # path to input .txt file which contains the estimated prior (pi)
    hmmemit = sys.argv[5] # path to input .txt file which contains the emission probabilities (B) 
    hmmtrans = sys.argv[6] # path to input .txt file which contains transition probabilities (A) 
    test_out = sys.argv[7] # path to the output .txt file to which the predicted tags will be written. 
                           # The file should be in the same format as the <test input> file
    metrics_out = sys.argv[8] # path of the output .txt file to write metrics

    # build and init 
    model = hmm_eval(index2word, index2tag, hmmprior, hmmemit, hmmtrans, metrics_out)

    # read the paramaters from files
    model.build_model()

    # testing: evaluate and write labels to output files
    log_lik, accuracy = model.evaluate(test_input, test_out, True)

    print("Average Log-Likelihood: ", log_lik, end=' ')
    print("Accuracy: ", accuracy)
    
    # Output: Metrics File
    with open(metrics_out, 'a') as f_metrics:
        f_metrics.write("Average Log-Likelihood: " + str(log_lik) + "\n")
        f_metrics.write("Accuracy: " + str(accuracy) + "\n")
