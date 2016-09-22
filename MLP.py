import numpy as np
from copy import copy
from iris_data_prepared import X_train, y_train, X_test, y_test
import matplotlib.pyplot as plt

class MLP():
    
    def __init__(self, train_data, train_labels, num_hidden_layers=2,
                    hidden_layer_size=50, activation="ReLU", l_rate=
                    0.01, reg=1e-4):
        
        # data
        self.X = np.array(train_data)
        self.y = np.ravel(np.array(train_labels))
        self.num_classes = len(np.unique(self.y))
        self.num_examples = self.X.shape[0]
        
        # structure
        self.num_hidden_layers = num_hidden_layers
        self.num_weight_layers = num_hidden_layers + 1
        self.hidden_layer_size = hidden_layer_size
        self.input_dimensions = self.X.shape[1]
        self.output_dimensions = 1
        self.initialise_weights()
        
        # bias
        self.initialise_biases()
        
        # activation
        self.activation = activation
        
        # hyperparameters
        self.l_rate = l_rate
        self.reg = reg

    def initialise_weights(self):
        self.weights = []
        self.weights.append(0.01*np.random.randn(self.input_dimensions,
                                            self.hidden_layer_size))
        for wl in range(self.num_weight_layers - 2):
            self.weights.append(0.01*np.random.randn(
                        self.hidden_layer_size, self.hidden_layer_size))
        self.weights.append(0.01*np.random.randn(self.hidden_layer_size,
                                                self.num_classes))
    
    def initialise_biases(self):
        self.biases = []
        self.biases.append(np.zeros((1, self.hidden_layer_size)))
        for b in range(self.num_weight_layers - 2):
            self.biases.append(np.zeros((1, self.hidden_layer_size)))
        self.biases.append(np.zeros((1, self.num_classes)))

    def forward_pass(self):
        self.activations = []
        self.activations.append(MLP.relu(np.dot(self.X, self.weights[0])
                                                      + self.biases[0]))
        for i in range(self.num_hidden_layers-1):
            self.activations.append(MLP.relu(np.dot(self.activations[i],
                                 self.weights[i+1]) + self.biases[i+1]))
        self.activations.append(MLP.softmax(np.dot(self.activations[-1],
                                   self.weights[-1]) + self.biases[-1]))
            
    def calc_loss(self):
        correct_logprobs = -np.log(self.activations[-1][range(
                                            self.num_examples), self.y])
        data_loss = np.sum(correct_logprobs) / self.num_examples
        reg_loss = 0.5 * self.reg * np.sum(self.weights[0] *
                                                    self.weights[0])
        self.loss = data_loss + reg_loss
    
    def back_propogation(self):
        # delta output scores
        layers = self.activations
        dscores = layers.pop()
        dscores[range(self.num_examples), self.y] -= 1
        dscores /= self.num_examples
        
        # delta weights and biases
        weights = copy(self.weights)
        biases = copy(self.biases)
        dweights, dbiases = self.get_deltas(layers, weights, biases,
                                                                dscores)

        # update weights
        for i in range(len(self.weights)):
            self.weights[i] += -self.l_rate * dweights[i]
        
        # update biases
        for i in range(len(self.biases)):
            self.biases[i] += -self.l_rate * dbiases[i]

    def get_deltas(self, layers, weights, biases, dscores):
        w = weights.pop()
        b = biases.pop()
        dweights = []
        dbiases = []
        d_incoming = dscores
        while len(weights) > 0:
            this_layer = layers.pop()
            dweights.append(np.dot(this_layer.T, d_incoming))
            dhl = np.dot(d_incoming, w.T)
            dhl[this_layer <= 0] = 0 # ReLU
            dbiases.append(np.sum(d_incoming, axis=0, keepdims=True))
            w = weights.pop()
            b = biases.pop()
            d_incoming = dhl
        dweights.append(np.dot(self.X.T, d_incoming))
        dbiases.append(np.sum(d_incoming, axis=0, keepdims=True))
        return dweights[::-1], dbiases[::-1]

    def predict(self, test_data, test_labels):
        activations = []
        activations.append(Neural_Network.relu(np.dot(self.X,
                                                self.weights[0])))
        for i in range(self.num_hidden_layers-1):
            activations.append(Neural_Network.relu(np.dot(
                            activations[i], self.weights[i+1])))
        
        activations.append(Neural_Network.softmax(
                            np.dot(activations[-1], self.weights[-1])))
    
        predicted_class = np.argmax(activations.pop(), axis=1)
        print(predicted_class)
    
    def train(self, iterations):
        x, y1, y2, y3 = [], [], [], []
        for i in range(iterations):
            self.forward_pass()
            self.calc_loss()
            self.back_propogation()
            if i % 1000 == 0:
                x.append(i)
                test = self.get_accuracy(X_train, y_train)
                train = self.get_accuracy(X_test, y_test)
                y1.append(test)
                y2.append(train)
                y3.append(self.loss)
                print(self.loss)
                print(test)
                print(train)
        print("accuracy on training set:",
                                    self.get_accuracy(X_train, y_train))
        print("accuracy on test set:",
                                    self.get_accuracy(X_test, y_test))
        print("error:", self.loss)
        MLP.visualise(x, y1, y2, y3)


    def get_accuracy(self, X, y):
        X = np.array(X)
        y = np.ravel(np.array(y))
        activations = []
        activations.append(MLP.relu(np.dot(X, self.weights[0])
                                                      + self.biases[0]))
        for i in range(self.num_hidden_layers-1):
            activations.append(MLP.relu(np.dot(activations[i],
                                 self.weights[i+1]) + self.biases[i+1]))
        activations.append(MLP.softmax(np.dot(activations[-1],
                                   self.weights[-1]) + self.biases[-1]))
        
        predicted_class = np.argmax(activations[-1], axis=1)
        return np.mean(predicted_class == y)


########################################################################
# activation functions
########################################################################

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x, deriv = False):
        if deriv == True: return x * (1 - x) 
        return 1.0 / (1.0 + np.exp(-x))    

    @staticmethod
    def tanh(x, deriv = False):
        if deriv == True: return 1.0 - np.tanh(x)**2
        return np.tanh(x)

    @staticmethod
    def softmax(scores):
        exp_scores = np.exp(scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

########################################################################
# visualisation
########################################################################

    @staticmethod
    def visualise(x, y1, y2, y3):
        plt.plot(x, y1)
        plt.plot(x, y2)
        plt.plot(x, y3)
        plt.show()

clf = MLP(X_train, y_train, 3, 25, "ReLU", 0.03, 1e-3)
clf.train(60000)
