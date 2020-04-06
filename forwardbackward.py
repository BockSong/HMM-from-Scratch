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
num_class = 10
epsilon = 1e-5
diff_th = 1e-7

class hmm_eval(object):
    def __init__(self, input_size, hidden_units, learning_rate, init_flag, metrics_out):
        self.learning_rate = learning_rate
        self.metrics_out = metrics_out
        self.layers = [
            linearLayer(input_size, hidden_units, init_flag),
            Sigmoid(),
            linearLayer(hidden_units, num_class, init_flag)
        ]
        self.criterion = softmaxCrossEntropy()

    def load_model(self):
        # open file and read parameters
        pass

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
    if len(sys.argv) != 6:
        print("The number of command parameters is incorrect.")
        exit(-1)

    test_input = sys.argv[1] # path to the test input .csv file
    index2word = sys.argv[2] # path to the .txt that specifies the dictionary mapping from words to indices. 
                             # The tags are ordered by index, with the first word having index of 1, the second word having index of 2, etc.
    index2tag = sys.argv[3] # path to the .txt that specifies the dictionary mapping from tags to indices. 
                            # The tags are ordered by index, with the first tag having index of 1, the second tag having index of 2, etc.
    train_out = sys.argv[3] # path to output .labels file which predicts on trainning data
    hmmprior = sys.argv[4] # path to output .txt file to which the estimated prior (π) will be written. 
    hmmemit = sys.argv[5] # path to output .txt file to which the emission probabilities (B) will be written. 
    hmmtrans = sys.argv[6] # path to output .txt file to which the transition probabilities (A) will be written. 
    test_out = sys.argv[7] #  path to output .labels file which predicts on test data
    metrics_out = sys.argv[8] # path of the output .txt file to write metrics

    # get input_size
    with open(train_input, 'r') as f_in:
        line = f_in.readline()
        split_line = line.strip().split(',')
        input_size = len(split_line) - 1

    # build and init 
    model = hmm_eval(input_size, hidden_units, learning_rate, init_flag, metrics_out)

    # read the paramaters from files
    model.load_model()

    # testing: evaluate and write labels to output files
    log_lik, accuracy = model.evaluate(train_input, train_out, True)

    print("Average Log-Likelihood: ", log_lik, end=' ')
    print("Accuracy: ", accuracy)
    
    # Output: Metrics File
    with open(metrics_out, 'a') as f_metrics:
        f_metrics.write("Average Log-Likelihood: " + str(log_lik) + "\n")
        f_metrics.write("Accuracy: " + str(accuracy) + "\n")
