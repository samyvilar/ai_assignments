__author__ = 'samyvilar'

from matplotlib import pyplot as plt
from multiprocessing import Process, Queue

import numpy


def update_progress(queue):
    plt.ion()
    input_set, output_set, epoch = queue.get()

    input_set, output_set = numpy.asarray(input_set).flatten(), numpy.asarray(output_set).flatten()
    plt.plot(input_set, output_set, 'b*-')
    line, = plt.plot(input_set, output_set, 'r*--')

    plt.show()
    while 1:
        input_set, output_set, epoch = queue.get()
        input_set, output_set = numpy.asarray(input_set).flatten(), numpy.asarray(output_set).flatten()
        line.set_data(input_set, output_set)
        plt.title('Epoch %i' % epoch)

        plt.draw()


class Animate_Progress(object):
    def __init__(self, input_normalization, output_normalization):
        self.input_normalization = input_normalization
        self.output = output_normalization
        self.queue = Queue()
        self.process = Process(target=update_progress, args=(self.queue,))
        self.process.start()
        self.queue.put((input_normalization, output_normalization, 0))
        self.epoch = 0
        self.epoch += 1

    def update(self, input_set, output_set):
        self.queue.put((input_set, output_set, self.epoch))
        self.epoch += 1

    def wait(self):
        while not self.queue.empty():
            pass
        self.process.terminate()

