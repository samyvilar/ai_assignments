__author__ = 'samyvilar'

from itertools import izip, starmap, imap, repeat, takewhile, tee, chain, ifilter
from collections import Iterable, deque

import numpy
from numpy import square, sqrt, multiply, add, zeros, dot
from numpy import ctypeslib

from multiprocessing import sharedctypes, Process, cpu_count


import activation
import normalization


number_of_cores = cpu_count()


class Layer(object):
    """
        We will assume each layer is fully connected to each subsequent layer, and contains a matrix W containing
        the weights of all those connections from the previous layer.
    """
    def __init__(
        self,
        number_of_neurons,
        number_of_inputs_per_neuron,
        activation_function,
        learning_rate=.2,
        momentum=.2,

    ):
        self.number_of_neurons = number_of_neurons
        self.number_of_inputs_per_neuron = number_of_inputs_per_neuron
        self.activation_function = getattr(activation, activation_function)()
        self.learning_rate = learning_rate

        self.weights, self.bias = self.activation_function.get_random_weights(
            (number_of_neurons, number_of_inputs_per_neuron)
        )
        self.momentum = momentum
        self.previous_weight_update = zeros(self.weights.shape)
        self.outputs = zeros(self.number_of_neurons)  # output vector

    def apply_input(self, normalized_inputs):
        """
            Apply input vector to layer, track current inputs/outputs discarding previous, return output vector.
            @normalized_inputs: a vector representing the inputs to this layer, numpy will apply its broadcasting rules
                                while applying the dot product
            @returns: activation_func(dot(weights, inputs) + bias)

            weights = numpy.random.random((number_of_neurons, number_of_inputs_per_neuron))  # Matrix
            inputs = numpy.random.random(number_of_neurons)  # vector
            bias = numpy.random.random(number_of_neurons) # vector
            output = activation_func(dot(weights, inputs) + bias) == activation_func((weights * inputs).sum(axis=1))
            output.shape == numpy.random.random(number_of_neurons).shape
            return output  # vector
        """
        self.inputs = normalized_inputs
        return self.activation_function.apply(
            add(dot(self.weights, self.inputs, self.outputs), self.bias, self.outputs), self.outputs
        )

    def update_weights(self, forward_layer_error_signal_factor):
        """
            Update the weights using gradient descent algorithm.
             @forward_layer_error_signal_factor: forward layers error factor as a vector
                ==> dot(forward_layer.weights.T, forward_layer.gradient)
        """
        # gradient_of_error_func = forward_layer_error_signal_factor * \
        #     self.activation_function.derivative(f_of_x=self.outputs) * self.learning_rate

        gradient_of_error_func = multiply(
            multiply(self.learning_rate, forward_layer_error_signal_factor, forward_layer_error_signal_factor),
            self.activation_function.derivative(f_of_x=self.outputs, out=self.outputs),
            self.outputs
        )

        back_propagation_error_factor = dot(self.weights.T, gradient_of_error_func)

        # delta_weights = (gradient_of_error_func.reshape((-1, 1)) * self.inputs) + \
        #                 (self.previous_weight_update * self.momentum)
        # self.previous_weight_update = delta_weights     # save delta weights for next cycle ...

        self.previous_weight_update = multiply(self.previous_weight_update, self.momentum, self.previous_weight_update)
        delta_weights = add(
            self.previous_weight_update,
            multiply(gradient_of_error_func.reshape((-1, 1)), self.inputs),
            self.previous_weight_update
        )

        self.weights += delta_weights                   # update weights.
        self.bias += gradient_of_error_func             # update bias.

        return back_propagation_error_factor            # back-propagate error factor to previous layer ...


class OutputLayer(Layer):
    pass


class HiddenLayer(Layer):
    pass


class InputLayer(Layer):
    def apply_input(self, normalized_input):
        self.inputs = normalized_input
        self.outputs = normalized_input  # we will rely on numpy broadcasting rules ...
        return self.outputs

    def update_weights(self, forward_layer_errors):  # The input layer has no (need to update) weights ...
        pass


class NeuralNetwork(object):
    def __init__(
            self,
            layers,
            allowed_error_threshold=0.00001,
            max_number_of_iterations=100000,
            normalization_class='MinMax'
    ):
        self.layers = layers
        self.allowed_error_threshold = allowed_error_threshold
        self.max_number_of_iterations = max_number_of_iterations
        self.normalization_class = getattr(normalization, normalization_class)
        self.layers_reversed = tuple(reversed(layers))

        sum_of_posterior_v = 1
        for layer in self.layers_reversed[:-1]:  # calculate learning rate for each layer ...
            v = (1.0 / layer.number_of_inputs_per_neuron) * sum_of_posterior_v
            layer.learning_rate /= (layer.number_of_inputs_per_neuron * sqrt(v))
            sum_of_posterior_v = numpy.sum(v)

    def _apply_input(self, network_inputs):
        return reduce(lambda neuron_inputs, layer: layer.apply_input(neuron_inputs), self.layers, network_inputs)

    def _update_weights(self, error):  # Update the weights by back propagating the error from output to input layer(s)
        _ = reduce(lambda _error_, l: l.update_weights(_error_), self.layers_reversed, error)
        return error

    # def _apply_samples(self, inputs, expected_outputs):  # update weights iteratively ('online training')
    #     return sqrt(
    #         sum(  # sum all the errors ...
    #             imap(
    #                 square,  # square the error, so they are all positive ...
    #                 imap(
    #                     self._update_weights,  # back propagate error ...
    #                     # get error ...                                              # apply input ...
    #                     (target - output for target, output in izip(expected_outputs, imap(self._apply_input, inputs)))
    #                 )
    #             )
    #         )
    #     )

    def _apply_samples(self, inputs, expected_outputs):  # update weights iteratively ('online training')
        return sqrt(
            sum(  # sum all the errors ...
                square(  # square the error, so they are all positive ...
                    map(
                        self._update_weights,  # back propagate error ...
                        # get error ...                                              # apply input ...
                        (target - output for target, output in izip(expected_outputs, imap(self._apply_input, inputs)))
                    )
                )
            )
        )

    def train(self, training_set, max_alloted_time=None):
        """ @Training_set: list of 2-tuple containing flatten input and corresponding output data set. """
        input_set, output_set = imap(numpy.asarray, izip(*training_set))  # convert training set to numpy array ...

        self.input_normalization, self.output_normalization = imap(self.normalization_class, (input_set, output_set))

        normalized_input_set = self.input_normalization.normalize(input_set)
        normalized_output_set = self.output_normalization.normalize(output_set)

        self.errors = tuple(
            takewhile(
                lambda sum_error, allowed_threshold=self.allowed_error_threshold: sum_error > allowed_threshold,
                starmap(
                    self._apply_samples,
                    repeat((normalized_input_set, normalized_output_set), self.max_number_of_iterations)
                )
            )
        )

    def feed_normalized_input(self, normalized_input):
        return self._apply_input(normalized_input)

    def feed(self, original_data_set):
        return self.output_normalization.de_normalize(
            self._apply_input(self.input_normalization.normalize(original_data_set))
        )