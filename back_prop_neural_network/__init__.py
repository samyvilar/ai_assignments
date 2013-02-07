__author__ = 'samy.vilar'
__date__ = '12/27/12'
__version__ = '0.0.1'


import urllib2
import datetime
import csv

from network import \
    Input_Layer, \
    Hidden_Layer, \
    Output_layer, \
    Neural_Network

def get_stock_history(stock_symbol):
    today_date = datetime.date.today()
    url = 'http://ichart.finance.yahoo.com/table.csv?s={symbol}&d={to_month_offset}&e={to_day}&f={to_year}&g=d&a={from_month_offset}&b={from_day}&c={from_year}&ignore=.csv'

    source = urllib2.urlopen(url.format(
        symbol = stock_symbol,
        to_month_offset = today_date.month - 1,
        to_day = today_date.day - 1,
        to_year = today_date.year,
        from_month_offset = 0,
        from_day = 1,
        from_year = 1900,
    ))

    return csv.DictReader(source)

def get_training_set(
            stock_symbol,
            history_length, # The number of consecutive days adj_close prices to return.
            starting_days,  # A list containing the start day index in reverse ie today(0), yesterday(1) and so on ...
    ):
    rows = [float(row['Adj Close'])
                for row in get_stock_history(stock_symbol)] # we only care about the closing price.

    return [rows[index:index + history_length + 1][::-1]
                for index in starting_days]


def get_trained_stock_neural_network(stock_prices_training_set):
    if not stock_prices_training_set:
        return None
    number_of_inputs = len(stock_prices_training_set[0]) - 1
    number_of_outputs = 1

    number_of_hidden_neurons = number_of_inputs + 20

    layers = [
        Input_Layer(
            number_of_inputs,                   # Number of Neurons
            1,                                  # Number of Inputs per Neuron
            number_of_hidden_neurons,           # Number of Outputs per Neuron
            'hyperbolic_tangent'),

        Hidden_Layer(
            number_of_hidden_neurons,
            number_of_inputs,
            number_of_outputs,
#            number_of_hidden_neurons,
            'hyperbolic_tangent',),

#        Hidden_Layer(
#            number_of_hidden_neurons,
#            number_of_hidden_neurons,
#            number_of_outputs,
#            'unipolar_sigmoid',),

        Output_layer(
            number_of_outputs,
            number_of_hidden_neurons,
            1,
            'hyperbolic_tangent',),
    ]

    neural_network = Neural_Network(layers = layers)
    neural_network.train(
        [(stock_price[:-1], stock_price[-1:])
            for stock_price in stock_prices_training_set]
    )

    return neural_network

if __name__ == '__main__':
    stock_symbol = 'aapl'
    number_of_consecutive_trading_days = 10
    trading_day_indices_from_today = [0]
    stock_prices = get_training_set(
        stock_symbol,
        number_of_consecutive_trading_days,
        trading_day_indices_from_today,)

    neural_network = get_trained_stock_neural_network(stock_prices)

    print "actual closing_price: {actual} predicted: {predicted}".format(
        actual = stock_prices[0][-1],
        predicted = neural_network.feed(stock_prices[0][:-1])
    )














 
