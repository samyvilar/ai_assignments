__author__ = 'samy.vilar'
__date__ = '12/27/12'
__version__ = '0.0.1'

from datetime import datetime
from itertools import compress, islice, imap, chain, izip

import numpy

from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from matplotlib import finance
from matplotlib import mlab

from network import InputLayer, HiddenLayer, OutputLayer, NeuralNetwork


def get_trained_stock_neural_network(stock_prices_training_set, number_of_hidden_neurons=30):
    number_of_inputs = len(stock_prices_training_set[0][0])
    number_of_outputs = len(stock_prices_training_set[0][1])

    layers = (
        InputLayer(
            number_of_inputs,                   # Number of Neurons
            1,                                  # Number of Inputs per Neuron
            'Identity'                          # Activation function.
        ),

        HiddenLayer(
            number_of_hidden_neurons,
            number_of_inputs,
            'HyperbolicTangent'
        ),

        OutputLayer(
            number_of_outputs,
            number_of_hidden_neurons,
            'Identity'                          # The activation function is giving the dot product, so do nothing ...
        )
    )

    neural_network = NeuralNetwork(
        layers=layers,
        allowed_error_threshold=.001,
        max_number_of_iterations=10000,
        normalization_class='Statistical'
    )
    neural_network.train(stock_prices_training_set)
    return neural_network


def test(neural_network, network_input, expected_output):
    print "actual closing_price: {actual} predicted: {predicted}".format(
        actual=expected_output,
        predicted=neural_network.feed(network_input)
    )


def train_as_function_interpolation(stock_data):
    neural_network = get_trained_stock_neural_network(
        [((float(index),), (stock_price,)) for index, stock_price in enumerate(stock_data)]
    )
    test(neural_network, len(stock_data) - 1, stock_data[-1])
    return neural_network


def sliding_window(values, window_size):
    return tuple(
        (values[index:(index + window_size)], values[(index + window_size):(index + window_size + 1)])
        for index in xrange(len(values) - window_size)
    )


def train_as_time_series(stock_prices, window_size):
    neural_network = get_trained_stock_neural_network(sliding_window(stock_prices, window_size))
    test(neural_network, stock_prices[-(window_size + 1):-1], stock_prices[-1])
    return neural_network


def plot(actual_data, predicted_data, end_of_training_point, x_axis, title='', errors=None):
    fig, axs = plt.subplots(2, sharex=False, sharey=False)

    actual_prices_plot = axs[0].plot_date(x_axis, actual_data, 'b*-')
    predicted_prices_plot = axs[0].plot_date(x_axis, predicted_data, 'r*--', label='predicted_prices')

    axs[0].set_ylabel("Adjusted Closing Price")
    axs[0].yaxis.set_major_formatter(plt.FormatStrFormatter('$%.2f'))
    axs[0].axvline(x=end_of_training_point, color='orange')

    errors_as_percentage = \
        ((numpy.asarray(actual_data) - numpy.asarray(predicted_data)) / numpy.asarray(actual_data)) * 100
    positive_values, negative_values = errors_as_percentage >= 0, errors_as_percentage < 0
    axs[1].bar(numpy.asarray(x_axis)[positive_values], errors_as_percentage[positive_values], color='g')
    axs[1].bar(numpy.asarray(x_axis)[negative_values], errors_as_percentage[negative_values], color='r')
    axs[1].axvline(x=end_of_training_point, color='orange')
    axs[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%3.3f%%'))

    axs[1].set_ylabel("difference as percentage")
    axs[0].set_xlabel("Trading Day")

    fig.autofmt_xdate()
    plt.title(title)
    ####################################################################################

    fig, ax = plt.subplots(1)
    actual_prices_plot = plt.plot(x_axis, actual_data, 'b*-')
    predicted_prices_plot = plt.plot(x_axis, predicted_data, 'r*--', label='predicted_prices')
    plt.axvline(x=end_of_training_point, color='orange')
    fig.autofmt_xdate()

    if errors:
        plt.figure()
        _ = plt.plot(xrange(len(errors)), errors, 'r--')

    plt.show()


def main():
    stock_symbol = 'nflx'
    number_of_training_days = 20
    window_size = 4
    percentage_of_trainable_days = .99

    stock_data = mlab.csv2rec(finance.fetch_historical_yahoo(stock_symbol, datetime(2013, 1, 1), datetime.today()))
    stock_prices = stock_data.adj_close[:number_of_training_days][::-1]
    dates = stock_data.date[:number_of_training_days][::-1]

    training_set = stock_prices[:int(number_of_training_days * percentage_of_trainable_days)]

    # neural_network = train_as_time_series(training_set, window_size)
    # network_samples = sliding_window(stock_prices, window_size)
    # predicted_prices = list(network_samples[0][0])
    # predicted_prices.extend(chain.from_iterable(neural_network.feed(inputs) for inputs, _ in network_samples))

    neural_network = train_as_function_interpolation(training_set)
    predicted_prices = [neural_network.feed(index)[0] for index in xrange(len(stock_prices))]

    plot(
        stock_prices,
        predicted_prices,
        dates[len(training_set) - 1],
        x_axis=dates,
        title='Trained as Function Interpolation',
        errors=neural_network.errors
    )


if __name__ == '__main__':
    main()
