__author__ = 'samyvilar'

import numpy
import pickle


def squared_error(exp, calc):
    return (exp - calc)**2

class Neuron(object):
    def __init__(self,
                 weights,
                 evaluation_func,
                 learning_rate = .5):

        self.weights = weights
        self.evaluation_func = evaluation_func
        self.learning_rate = learning_rate

    def calc_output(self, inputs):
        self.inputs = inputs
        self.output = self.evaluation_func(numpy.dot(self.inputs, self.weights))
        return self.output


def normalize(data_set, data_set_mean, data_set_max): # normalize around 0
    return (data_set - data_set_mean)/data_set_max
def de_normalize(normalize_data_set, data_set_mean, data_set_max):
    return (normalize_data_set * data_set_max) + data_set_mean


class Layer(object):
    def __init__(self,
                 number_of_neurons,
                 number_of_inputs_per_neuron,
                 number_of_outputs_per_neuron,
                 evaluation_function
    ):     # Add an extra weight for the bias '1'
        # number_of_inputs_per_neuron += 1 #TODO: Implement Bias, to better work with zero inputs.

        self.weights = numpy.random.random((number_of_neurons, number_of_inputs_per_neuron)) - 0.5
        self.neurons = [Neuron(
                    self.weights[index],
                    evaluation_function,
                )
            for index in xrange(number_of_neurons)
        ]
        self.number_of_outputs_per_neuron = number_of_outputs_per_neuron
class Input_Layer(Layer):
    def apply_input(self, normalized_input):
        self.inputs = normalized_input
        self.outputs = normalized_input
        return self.outputs
    def update_weights(self, forward_layer_errors):
        pass

class Hidden_Layer(Layer):
    def apply_input(self, normalized_inputs):
        self.inputs = normalized_inputs
        self.outputs = numpy.asarray(
            [neuron.calc_output(normalized_inputs)
                    for index, neuron in enumerate(self.neurons)]
        )
        return self.outputs

    def update_weights(self, forward_layer_errors):
        gradient_errors = self.outputs * (1 - self.outputs) * forward_layer_errors
        self.weights += gradient_errors.reshape(gradient_errors.size, 1) * self.inputs.reshape(1, self.inputs.size)
        for index, neuron in enumerate(self.neurons):
            neuron.weights = self.weights[index]
        return (gradient_errors * self.weights.T).T.sum(axis = 0)


class Output_layer(Layer):
    def apply_input(self, normalized_inputs):
        self.inputs = normalized_inputs
        self.outputs = numpy.asarray(
                [neuron.calc_output(normalized_inputs)
                    for neuron in self.neurons])
        return self.outputs

    def update_weights(self, forward_layer_errors):
        target_set, output_set = forward_layer_errors
        gradient_error = (target_set - output_set) * output_set * (1 - output_set)

        self.weights += gradient_error * self.inputs
        for index, neuron in enumerate(self.neurons):
            neuron.weights = self.weights[index]
        return (self.weights * gradient_error).sum(axis = 1)

class Neural_Network(object):
    def __init__(self, layers = []):
        self.layers = layers
    @property
    def input_layer(self):
        return self.layers[0]
    @property
    def output_layer(self):
        return self.layers[-1]
    @property
    def hidden_layers(self):
        return self.layers[1:-1]

    def _apply_input(self, normalize_input_set):
        prev_layer_outputs = normalize_input_set
        for layer in self.layers:
            prev_layer_outputs = layer.apply_input(prev_layer_outputs)
        return prev_layer_outputs

    def apply_sample(self, normalized_training_sample):
        input_set, target_set = normalized_training_sample[0], normalized_training_sample[1]
        prev_layer_outputs = self._apply_input(input_set)

        forward_layer_errors = (target_set,  prev_layer_outputs)
        for layer in reversed(self.layers):
            forward_layer_errors = layer.update_weights(forward_layer_errors)

        return (target_set - prev_layer_outputs)**2

    def train(self, training_set, max_number_of_iterations = 100000, allowed_error_threshold = 0.001):
        complete_data_set = numpy.asarray([input_set + output_set for input_set, output_set in training_set])
        self.original_mean, self.original_max = complete_data_set.mean(), complete_data_set.max()

        normalize_complete_set = normalize(complete_data_set, self.original_mean, self.original_max)
        normalized_input_training_set = normalize_complete_set[:, :len(training_set[0][0])]
        normalize_output_training_set = normalize_complete_set[:, len(training_set[0][0]):]

        errors = []
        for iteration_index in xrange(max_number_of_iterations):
            errors.append(
                numpy.sum(
                    self.apply_sample((normalized_input_set, normalize_output_training_set[index]))
                        for index, normalized_input_set in enumerate(normalized_input_training_set)
                )
            )
            if errors[-1] < allowed_error_threshold:
                break
            print "iteration: {iteration}, sum_of_errors: {sum_of_errors}".format(
                iteration = iteration_index,
                sum_of_errors = errors[-1]
            )

    def feed(self, original_data_set):
        normalize_output_data_set = self._apply_input(
            normalize(
                original_data_set,
                self.original_mean,
                self.original_max
            )
        )
        return de_normalize(
            normalize_output_data_set,
            self.original_mean,
            self.original_max
        )
