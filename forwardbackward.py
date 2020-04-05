'''

forwardbackward.py

Implementation of hidden Markov model.

Forward backward algorithm.

'''

import sys
import math
import numpy as np

Debug = True
PrintGrad = True
num_class = 10
epsilon = 1e-5
diff_th = 1e-7

class hmm(object):
    def __init__(self, input_size, hidden_units, learning_rate, init_flag, metrics_out):
        self.learning_rate = learning_rate
        self.metrics_out = metrics_out
        self.layers = [
            linearLayer(input_size, hidden_units, init_flag),
            Sigmoid(),
            linearLayer(hidden_units, num_class, init_flag)
        ]
        self.criterion = softmaxCrossEntropy()

    # SGD_step: update params by taking one SGD step
    # <x> a 1-D numpy array
    # <y> an integer within [0, num_class - 1]
    def SGD_step(self, x, y):
        # perform forward propogation and compute intermediate results
        if PrintGrad:
            print("		Begin forward pass")
        for layer in self.layers:
            x = layer.forward(x)
            if PrintGrad:
                print("     output: ", x)
        loss, _ = self.criterion.forward(x, y)
        if PrintGrad:
            print("			Cross entropy: ", loss)
            print("		Begin backward pass")

        # perform back propagation and update parameters
        delta = self.criterion.backward()
        if PrintGrad:
            print("			d(loss)/d(softmax inputs): ", delta)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate)
            if PrintGrad:
                print("     delta: ", delta)

        if PrintGrad:
            print("			New first layer weights: ", self.layers[0].W)
            print("			New first layer bias: ", self.layers[0].b)
            print("			New second layer weights: ", self.layers[2].W)
            print("			New second layer bias: ", self.layers[2].b)
        return loss


    def train_model(self, train_file, num_epoch):
        dataset = [] # a list of features
        # read the dataset
        with open(train_file, 'r') as f:
            for line in f:
                split_line = line.strip().split(',')
                y = int(split_line[0])
                x = np.asarray(split_line[1:], dtype=int)
                #feature[len(self.dic)] = 1 # add the bias feature
                dataset.append([y, x])

        with open(metrics_out, 'w') as f_metrics:
            # perform training
            for epoch in range(num_epoch):
                loss = 0
                for idx in range(len(dataset)):
                    loss = self.SGD_step(dataset[idx][1], dataset[idx][0])
                    if Debug and (idx % 1000 == 0):
                        print("[Epoch ", epoch + 1, "] Step ", idx + 1, ", current_loss: ", loss)

                train_loss, train_error = self.evaluate(train_input, train_out)
                test_loss, test_error = self.evaluate(test_input, test_out)

                if Debug:
                    print("[Epoch ", epoch + 1, "] ", end='')
                    print("train_loss: ", train_loss, end=' ')
                    print("train_error: ", train_error)
                    print("test_loss: ", test_loss, end=' ')
                    print("test_error: ", test_error)

                f_metrics.write("epoch=" + str(epoch) + " crossentryopy(train): " + str(train_loss) + "\n")
                f_metrics.write("epoch=" + str(epoch) + " crossentryopy(test): " + str(test_loss) + "\n")

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
                           # The file output to this path should be in the same format as the handout hmmprior.txt
    hmmemit = sys.argv[5] # path to output .txt file to which the emission probabilities (B) will be written. 
                          # The file output to this path should be in the same format as the handout hmmemit.txt
    hmmtrans = sys.argv[6] # path to output .txt file to which the transition probabilities (A) will be written. 
                           # The file output to this path should be in the same format as the handout hmmtrans.txt
    test_out = sys.argv[7] #  path to output .labels file which predicts on test data
    metrics_out = sys.argv[8] # path of the output .txt file to write metrics

    # get input_size
    with open(train_input, 'r') as f_in:
        line = f_in.readline()
        split_line = line.strip().split(',')
        input_size = len(split_line) - 1

    # build and init 
    model = hmm(input_size, hidden_units, learning_rate, init_flag, metrics_out)

    # training
    model.train_model(train_input, num_epoch)

    # testing: evaluate and write labels to output files
    train_loss, train_error = model.evaluate(train_input, train_out, True)
    test_loss, test_error = model.evaluate(test_input, test_out, True)

    print("train_loss: ", train_loss, end=' ')
    print("train_error: ", train_error)
    print("test_loss: ", test_loss, end=' ')
    print("test_error: ", test_error)
    
    # Output: Metrics File
    with open(metrics_out, 'a') as f_metrics:
        f_metrics.write("Average Log-Likelihood: " + str(train_error) + "\n")
        f_metrics.write("Accuracy: " + str(test_error) + "\n")
