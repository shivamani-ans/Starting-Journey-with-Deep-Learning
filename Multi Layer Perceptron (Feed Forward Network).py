#https://iamtrask.github.io/2015/07/12/basic-python-network/ <need to implement>
#https://github.com/miloharper/multi-layer-neural-network/blob/master/main.py

from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        random.seed(1)

        self.synaptic_weights = 2 * random.random((3,1)) - 1
        # syn0 = 2*np.random.random((3,4)) - 1
        # syn1 = 2*np.random.random((4,1)) - 1
    
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))
    
    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))
    
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output = self.think(training_set_inputs)
            error = training_set_outputs - output
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output)) #why product of MM(inputs, error*sigmoid_derivative(output))
            self.synaptic_weights += adjustment
    
if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    training_set_outputs = array([[0,1,1,0]]).T

    neural_network.train(training_set_inputs,training_set_outputs,10000)
    print("New synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    print("Considering new situation [1,0,0] -> ?: ")
    print(neural_network.think(array([1,0,0])))

## 2-Layer feed forward neural networks
# l0 = X
# l1 = nonlin(np.dot(l0,syn0))
# l2 = nonlin(np.dot(l1,syn1))
 
# # how much did we miss the target value?
# l2_error = y - l2

# if (j% 10000) == 0:
#     print "Error:" + str(np.mean(np.abs(l2_error)))

# # in what direction is the target value?
# # were we really sure? if so, don't change too much.

# l2_delta = l2_error*nonlin(l2,deriv=True)

# # how much did each l1 value contribute to the l2 error (according to the weights)?

# l1_error = l2_delta.dot(syn1.T)

# # in what direction is the target l1?
# # were we really sure? if so, don't change too much.

# l1_delta = l1_error * nonlin(l1,deriv=True)

# syn1 += l1.T.dot(l2_delta)
# syn0 += l0.T.dot(l1_delta)
