__author__ = 'samyvilar'

import numpy

class unipolar_sigmoid(object):
    def apply(self, value):
        return 1.0/(1.0 + (numpy.e**(-value)))
    def derivative(self, value = None, f_of_x = None):
        if value is not None:
            f_of_x = self.apply(value)
        return (1.0 - f_of_x) * f_of_x
    def inverse(self, output):
        return numpy.log(output/(1 - output))

class bipolar_sigmoid(object):
    def apply(self, value):
        return -1.0 + (2.0/(1.0 + numpy.e**(-value)))
    def derivative(self, value = None, f_of_x = None):
        if value is not None:
            f_of_x = self.apply(value)
        return 0.5 * (1.0 + f_of_x)*(1.0 - f_of_x)

class hyperbolic_tangent(object):
    def apply(self, value):
        return (numpy.e**value - numpy.e**(-value))/(numpy.e**value + numpy.e**(-value))
    def derivative(self, value = None, f_of_x = None):
        if value is not None:
            f_of_x = self.apply(value)
        return 1.0 - f_of_x**2


def squared_error(exp, calc):
    return (exp - calc)**2

def normalize(data_set, data_set_mean, data_set_max): # normalize around 0
    return (data_set - data_set_mean)/data_set_max
def de_normalize(normalize_data_set, data_set_mean, data_set_max):
    return (normalize_data_set * data_set_max) + data_set_mean


class Layer(object):
    def __init__(self,
                 number_of_neurons,
                 number_of_inputs_per_neuron,
                 number_of_outputs_per_neuron,
                 activation_function,
                 learning_rate = 1.0
    ):
        #TODO: Implement Bias, to better work with zero inputs.

        self.weights = numpy.random.random((number_of_neurons, number_of_inputs_per_neuron)) - 0.5
        self.number_of_outputs_per_neuron = number_of_outputs_per_neuron
        self.number_of_inputs_per_neuron = number_of_inputs_per_neuron
        self.activation_function = globals()[activation_function]()

        self.learning_rate = learning_rate

    def _update_weights(self, deltas):
        self.weights += self.learning_rate * deltas

    def _calc_gradient_error(self, forward_layer_errors):
        return self.activation_function.derivative(f_of_x = self.outputs) * forward_layer_errors

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
        self.outputs = self.activation_function.apply(
            numpy.dot(self.weights, normalized_inputs)
        )
        return self.outputs

    def update_weights(self, forward_layer_errors):
        gradient_errors = self._calc_gradient_error(forward_layer_errors)
        self._update_weights(
            numpy.repeat(
                            self.inputs.reshape(1, self.inputs.shape[0]),
                            self.weights.shape[0],
                            axis = 0,
                    ) * gradient_errors.reshape(gradient_errors.shape[0], 1)
        )
        return (gradient_errors * self.weights.T).sum(axis = 0)


class Output_layer(Layer):
    def apply_input(self, normalized_inputs):
        self.inputs = normalized_inputs
        self.outputs = self.activation_function.apply(
            numpy.dot(self.weights, normalized_inputs)
        )
        return self.outputs
    def update_weights(self, forward_layer_errors):
        gradient_errors = self._calc_gradient_error(forward_layer_errors)
        self._update_weights(self.inputs * gradient_errors)   # Adjust the weights based on the error

        return (gradient_errors * self.weights.T).sum(axis = 0) # back-propagate errors to previous layers.

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

        forward_layer_errors = target_set -  prev_layer_outputs
        for layer in reversed(self.layers):
            forward_layer_errors = layer.update_weights(forward_layer_errors)

        return target_set - prev_layer_outputs

    def train(self, training_set, max_number_of_iterations = 100000, allowed_error_threshold = 0.0001):
        # Training_set must be a list of 2-tuple containing flatten input and corresponding output data set.
        complete_data_set = numpy.asarray([input_set + output_set for input_set, output_set in training_set])
        self.original_mean, self.original_max = complete_data_set.mean(), complete_data_set.max()

        normalize_complete_set = normalize(complete_data_set, self.original_mean, self.original_max)
        normalized_input_training_set = normalize_complete_set[:, :len(training_set[0][0])]
        normalize_output_training_set = normalize_complete_set[:, len(training_set[0][0]):]

        errors = []
        for iteration_index in xrange(max_number_of_iterations):
            errors.append(
                 0.5 * numpy.sum(
                    self.apply_sample((normalized_input_set, normalize_output_training_set[index]))**2
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
