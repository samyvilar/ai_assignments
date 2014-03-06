__author__ = 'samyvilar'

from matplotlib import pyplot as plt
from multiprocessing import Process, Queue
from types import MethodType

import numpy


def initial_plot(input_set, output_set, epoch, relative_error):
    plt.ion()
    input_set, output_set = numpy.asarray(input_set).flatten(), numpy.asarray(output_set).flatten()
    plt.plot(input_set, output_set, 'b*-')
    line, = plt.plot(input_set, output_set, 'r*--')
    plt.show()
    return line


def update_plot(line, input_set, output_set, epoch, relative_error):
    line.set_ydata(output_set)
    plt.title('Epoch %i, Relative Error %f' % (epoch, relative_error))
    plt.draw()


def update_progress(queue):
    # input_set, output_set, epoch, relative_error = queue.get()
    line = initial_plot(*queue.get())
    while 1:
        update_plot(line, *queue.get())
        # input_set, output_set, epoch, relative_error = queue.get()
        # input_set, output_set = numpy.asarray(input_set).flatten(), numpy.asarray(output_set).flatten()
        # line.set_ydata(output_set)
        # plt.title('Epoch %i, Relative Error %f' % (epoch, relative_error))
        # plt.draw()


class AnimatePlots(object):
    def __init__(self, use_process=True):
        self.queue = Queue()
        if use_process:
            self.process = Process(target=update_progress, args=(self.queue,))
            self.process.start()
        else:
            class Plot(object):
                def initialize_line(self, *args):
                    self.line = initial_plot(*args)
                    self.get_line = MethodType(lambda self, *args: self.line, self)
                    return self.line

                def put(self, args):
                    update_plot(getattr(self, 'get_line', self.initialize_line)(*args), *args)

                def empty(self):
                    pass
            self.process = type('NoProcess', (object,), {'terminate': lambda: None, 'start': lambda: None})()
            self.queue = Plot()

    def update(self, *args):
        self.queue.put(args)

    def stop(self):
        while not self.queue.empty():
            pass
        self.process.terminate()

