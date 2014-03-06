__author__ = 'samyvilar'

from itertools import starmap, repeat, izip, chain, imap
from collections import defaultdict

import random
import numpy
import hashlib

import os
from ctypes import CDLL, POINTER, c_ubyte, c_uint, c_void_p


def diagonals(a):
    rows, cols = a.shape
    # if cols > rows:  assume that cols == rows ...
    #     a = a.T
    #     rows, cols = a.shape
    fill = numpy.zeros(((cols - 1), cols), dtype=a.dtype)
    stacked = numpy.vstack((a, fill, a))
    major_stride, minor_stride = stacked.strides
    strides = major_stride, minor_stride * (cols + 1)
    shape = (rows + cols - 1, cols)
    return numpy.lib.stride_tricks.as_strided(stacked, shape, strides)


def py_diagonal_collisions(solution):
    column_axis, row_axis = 0, 1
    left_diagonal_queens = diagonals(solution).sum(axis=row_axis)
    right_diagonal_queens = diagonals(numpy.fliplr(solution)).sum(axis=row_axis)
    return left_diagonal_queens[numpy.where(left_diagonal_queens > 1)].sum() + \
        right_diagonal_queens[numpy.where(right_diagonal_queens > 1)].sum()


try:
    libo = CDLL(os.path.join(os.path.dirname(__file__), 'libcollisions.so'))
    libo.collisions.argtypes = [c_void_p, c_uint]
    libo.collisions.restype = c_uint

    def diagonal_collisions(solution):
        assert solution.dtype == numpy.uint8
        return libo.collisions(c_void_p(solution.ctypes.data), c_uint(solution.shape[0]))
except OSError as _:
    diagonal_collisions = py_diagonal_collisions


# make sure to properly set the cache size, depending on the the system
# default is 3 gigs.
def calc_collision(solution, cache=defaultdict(defaultdict), cache_size_bytes=3 * 10**9):
    hash_value = hashlib.sha1(solution).hexdigest()

    if hash_value in cache:
        # assert not numpy.sum(cache[hash]['solution'] - solution) # TODO: check for collisions.
        return cache[hash]['diagonal_collision']

    diag_cols = diagonal_collisions(solution)
    #since we are shuffling the columns their will never be any collision around the rows or columns.
    #queens_in_rows = solution.sum(axis = row_axis)
    #row_collision = queens_in_rows[numpy.where(queens_in_rows > 1)].sum()
    #queens_in_columns = solution.sum(axis = column_axis)
    #column_collision = queens_in_columns[numpy.where(queens_in_columns > 1)].sum()

    # get the collisions around the left and right diagonals ...


    #if sys.getsizeof(cache) < cache_size_bytes:
    cache[hash]['diagonal_collision'] = diag_cols
    #cache[hash]['solution'] = solution # TODO: check for collisions.

    return diag_cols


def new_initialized_board(columns):
    return numpy.vstack(tuple(starmap(numpy.eye, izip(repeat(1), repeat(len(columns)), columns))))


def irow(length, set_column=0):
    return chain(repeat(0, set_column), (1,), repeat(0, length - set_column - 1))


def irandom_board(number_of_queens):
    return chain.from_iterable(
        imap(irow, repeat(number_of_queens), random.sample(xrange(number_of_queens), number_of_queens))
    )


def new_random_board(number_of_queens):
    board = numpy.identity(number_of_queens, dtype='ubyte')
    numpy.random.shuffle(board)  # randomly shuffle the rows ...
    return board


def get_set_columns(board):
    return numpy.where(board == 1)[1]


def sample_population(population, sample_size):
    random_indices = numpy.arange(len(population))
    numpy.random.shuffle(random_indices)
    samples = population[random_indices[:sample_size]]
    return samples[0] if sample_size == 1 else samples