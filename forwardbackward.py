'''

forwardbackward.py

Implementation of hidden Markov model.

Run the forward backward algorithm.

'''

import sys
import math
import numpy as np

Debug = False

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
        # read the paramaters from files
        self.build_model()

    # open file and read parameters
    def build_model(self):
        # load word2idx
        with open(self.index2word, 'r') as f_idx2word:
            i = 0
            for line in f_idx2word:
                line = line.replace('\n', '').replace('\r', '')
                self.word2idx[line] = i
                i += 1

        # load tag2idx
        with open(self.index2tag, 'r') as f_index2tag:
            i = 0
            for line in f_index2tag:
                line = line.replace('\n', '').replace('\r', '')
                self.tag2idx[line] = i
                i += 1

        if Debug:
            print(self.word2idx)
            print(self.tag2idx)

        # parameter initialization
        self.N = len(self.tag2idx) # number of tags (candidate observations)
        self.M = len(self.word2idx) # number of words (candidate states)

        self.pi = np.zeros(self.N)
        self.A = np.zeros((self.N, self.N))
        self.B = np.zeros((self.N, self.M))

        # read the parameters
        with open(self.hmmprior, 'r') as f_prior:
            for j in range(self.N):
                num = f_prior.readline().strip()
                self.pi[j] = float(num)
        
        with open(self.hmmtrans, 'r') as f_emit:
            for j in range(self.N):
                words = f_emit.readline().strip().split(' ')
                for k in range(self.N):
                    self.A[j][k] = float(words[k])

        with open(self.hmmemit, 'r') as f_trans:
            for j in range(self.N):
                words = f_trans.readline().strip().split(' ')
                for k in range(self.M):
                    self.B[j][k] = float(words[k])
            
        # log-space
        self.pi = np.log(self.pi)
        self.A = np.log(self.A)
        self.B = np.log(self.B)

        if Debug:
            print(self.pi)
            print(self.A)
            print(self.B)

    # given x, compute the forward prob predict
    def forward(self, x):
        T = x.size
        self.alpha = np.zeros((T, self.N))

        # log-space
        #for j in range(self.N):
        #    self.alpha[0][j] = self.pi[j] + self.B[j][x[0]]
        self.alpha[0] = self.pi + self.B.T[x[0]]

        for t in range(1, T):
            for j in range(self.N):
                sum_exp = 0
                # log-sum-exp trick
                v = self.alpha[t - 1] + self.A.T[j]
                m = np.max(v)
                for k in range(self.N):
                    #sum_exp += np.exp(self.alpha[t - 1][k] + self.A[k][j])
                    sum_exp += np.exp(v[k] - m)
                self.alpha[t][j] = self.B[j][x[t]] + m + np.log(sum_exp)

    # given x, compute the backward prob predict
    def backward(self, x):
        T = x.size
        self.beta = np.zeros((T, self.N))

        # log-space
        #for j in range(self.N):
        #    self.beta[T - 1][j] = 1
        # just 0

        for t in range(T - 2, -1, -1):
            for j in range(self.N):
                sum_exp = 0
                # log-sum-exp trick
                v = self.B.T[x[t + 1]] + self.beta[t + 1] + self.A[j]
                m = np.max(v)
                for k in range(self.N):
                    sum_exp += np.exp(v[k] - m)
                self.beta[t][j] = m + np.log(sum_exp)

    # make pred to each word in sequence x, by Minimum Bayes Risk Prediction
    def predict(self, x):
        # need to go back to normal space rather than using log(alpha or beta)
        cond_prob = np.multiply(np.exp(self.alpha), np.exp(self.beta))
        return np.argmax(cond_prob, axis=1)

    # compute avg log-lik using f-b algo and predict tags using MBR
    def evaluate(self, in_path, out_path, write = False):
        log_lik, corr_word, word_count, sent_count = 0., 0., 0., 0.

        with open(in_path, 'r') as f_in:
            with open(out_path, 'w') as f_out:
                for line in f_in:
                    words = line.strip().split(' ')
                    x_str, x, y_str, y = [], [], [], []
                    for ele in words:
                        word = ele.split('_')
                        x_str.append(word[0])
                        y_str.append(word[1])
                        x.append(self.word2idx[word[0]])
                        y.append(self.tag2idx[word[1]])
                    
                    # transfer from list to numpy array
                    x = np.array(x)
                    y = np.array(y)
                    T = x.size

                    # run forward and backward
                    self.forward(x)
                    self.backward(x)

                    # compute the log likehood of the sequence
                    # need to first go back to normal space and then log to the summation
                    this_log_lik = np.log(np.sum(np.exp(self.alpha[T - 1])))
                    log_lik += this_log_lik

                    tag_key = list(self.tag2idx.keys())
                    tag_val = list(self.tag2idx.values())
                    
                    pred = self.predict(x)
                    f_out.write(str(x_str[0]) + "_" + tag_key[tag_val.index(pred[0])])
                    for i in range(1, T):
                        f_out.write(" " + str(x_str[i]) + "_" + tag_key[tag_val.index(pred[i])])
                    f_out.write("\n")

                    # accuracy should be word-level
                    if Debug:
                        print("pred: ", pred)
                        print("y: ", y)
                    corr_word += np.sum(pred == y)

                    word_count += T
                    sent_count += 1

        return log_lik / sent_count, corr_word / word_count


if __name__ == '__main__':
    if len(sys.argv) != 9:
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

    # testing: evaluate and write labels to output files
    avg_log_lik, accuracy = model.evaluate(test_input, test_out, True)

    print("Average Log-Likelihood: ", avg_log_lik)
    print("Accuracy: ", accuracy)
    
    # Output: Metrics File
    with open(metrics_out, 'w') as f_metrics:
        f_metrics.write("Average Log-Likelihood: " + str(avg_log_lik) + "\n")
        f_metrics.write("Accuracy: " + str(accuracy) + "\n")
