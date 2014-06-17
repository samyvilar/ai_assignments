__author__ = 'samyvilar'

from itertools import starmap, repeat, izip, chain, imap, product, count, ifilter
from collections import defaultdict, OrderedDict, deque

from inspect import isgenerator, isgeneratorfunction
import random
import math
import numpy
import hashlib
import logging

import os
from ctypes import CDLL, POINTER, c_ubyte, c_uint, c_void_p

board_element_type = 'uint32'


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


# def py_diagonal_collisions(solution):
#     column_axis, row_axis = 0, 1
#     left_diagonal_queens = diagonals(solution).sum(axis=row_axis)
#     right_diagonal_queens = diagonals(numpy.fliplr(solution)).sum(axis=row_axis)
#     return left_diagonal_queens[numpy.where(left_diagonal_queens > 1)].sum() + \
#         right_diagonal_queens[numpy.where(right_diagonal_queens > 1)].sum()

# a diagonal collision occurs between two queens if abs(delta(q_xcords)) == abs(delta(q_ycords))
# assuming theres no queen between them ...
# 1) a naive solution would simply iterate over all the n^2 possible locations adding up collisions ignoring if
# there's any queens in between
# 2) go over each diagnol and count the number of queens there,
# the number of collision per diagonal is min(number of queens, 1) - 1
def py_diagonal_collisions(perm):
    return (
        sum(
            imap(
                numpy.equal,
                imap(numpy.abs, starmap(numpy.subtract, product(*repeat(numpy.arange(len(perm)), 2)))),
                imap(numpy.abs, starmap(numpy.subtract, product(*repeat(perm, 2))))
            )
        ) - len(perm)  # subtract n queens since since a queen is always colliding with itself ...
    )  # divide by 2 since the attacks are symmetrical ...


    # return numpy.arange(len(col_cords)) == numpy.abs()
    # colls_cnt = 0
    # for queen_a in xrange(len(col_cords)):
    #     for queen_b in xrange(len(col_cords)):
    #         colls_cnt += abs(queen_a - queen_b) == abs(col_cords[queen_a] - col_cords[queen_b])
    # return colls_cnt


def ipermutation_inversion(perm):  # get the inverse permutation ...
    perm = numpy.asarray(perm)    # for each index sum all the values preceding it in the permutation which are greater
    return imap(numpy.sum, ((perm[:index] > perm[index]) for index in numpy.argsort(perm)))


def py_permutation_from_inversion(inv_seq):  # get a permutation from its inverse sequence ...
    inv_seq = numpy.asarray(inv_seq if hasattr(inv_seq, '__getitem__') else tuple(inv_seq), dtype=board_element_type)
    pos = inv_seq.copy()
    for i in reversed(xrange(len(inv_seq))):
        pos[i + 1:] += pos[i + 1:] >= inv_seq[i]
    return numpy.argsort(pos).astype(board_element_type)


try:
    libo = CDLL(os.path.join(os.path.dirname(__file__), 'libutils.so'))

    libo.diag_collisions.argtypes = [c_void_p, c_uint]
    libo.diag_collisions.restype = c_uint

    libo.permutation_inversion.argtypes = [c_void_p, c_void_p, c_uint]
    libo.perm_from_inv_seq.argtypes = libo.permutation_inversion.argtypes

    def diagonal_collisions(perm):  # return the number of collisions on the diagonals of
        assert perm.dtype == numpy.uint32
        return libo.diag_collisions(c_void_p(perm.ctypes.data), c_uint(len(perm)))

    def permutation_inversion(perm):
        assert perm.dtype == numpy.uint32
        result = numpy.empty(len(perm), dtype=perm.dtype)
        libo.permutation_inversion(c_void_p(perm.ctypes.data), c_void_p(result.ctypes.data), c_uint(len(perm)))
        return result

    def permutation_from_inversion(inv_seq):
        assert inv_seq.dtype == numpy.uint32
        result = numpy.empty(len(inv_seq), dtype=inv_seq.dtype)
        libo.perm_from_inv_seq(c_void_p(inv_seq.ctypes.data), c_void_p(result.ctypes.data), c_uint(len(inv_seq)))
        return result

except OSError as er:
    logging.warning('Failed to load corresponding C implementations: {0}'.format(er))
    diagonal_collisions = py_diagonal_collisions
    permutation_inversion = ipermutation_inversion
    permutation_from_inversion = py_permutation_from_inversion


# make sure to properly set the cache size, depending on the the system
# default is 3 gigs.
def calc_collision(solution, cache=defaultdict(defaultdict), cache_size_bytes=3 * 10**9):
    hash_value = hashlib.sha1(solution).hexdigest()

    if hash_value in cache:
        # assert not numpy.sum(cache[hash]['solution'] - solution) # TODO: check for collisions.
        return cache[hash]['diagonal_collision']

    return diagonal_collisions(solution)
    #since we are shuffling the columns their will never be any collision around the rows or columns.

    #queens_in_rows = solution.sum(axis = row_axis)
    #row_sums = solution.sum(1)
    #column_sums = solution.sum(0)
    #collisions = numpy.sum(row_sums[row_sums > 1] - 1) + numpy.sum(column_sums[column_sums > 1] - 1)
    #queens_in_columns = solution.sum(axis = column_axis)
    #column_collision = queens_in_columns[numpy.where(queens_in_columns > 1)].sum()

    # get the collisions around the left and right diagonals ...


    #if sys.getsizeof(cache) < cache_size_bytes:
    #cache[hash]['diagonal_collision'] = diag_cols
    #cache[hash]['solution'] = solution # TODO: check for collisions.

    #return diag_cols


def irow(length, set_column=0):
    return chain(repeat(0, set_column), (1,), repeat(0, length - set_column - 1))


def irandom_board(number_of_queens):
    return chain.from_iterable(imap(irow, repeat(number_of_queens), sample(xrange(number_of_queens), number_of_queens)))


def new_random_board(number_of_queens):
    col_indices = numpy.arange(number_of_queens, dtype=board_element_type)
    numpy.random.shuffle(col_indices)
    return col_indices
    # board = numpy.identity(number_of_queens, dtype=board_element_type)
    # numpy.random.shuffle(board)  # randomly shuffle the rows ...
    # return board


def new_initialized_board(perm, dtype=board_element_type):  # create n by n matrix containing 1 at row i, column perm[i]
    return numpy.fromiter(
        chain.from_iterable(imap(irow, repeat(len(perm)), perm)), dtype=board_element_type, count=len(perm)**2
    ).reshape((len(perm), len(perm)))


def columns(board):
    return numpy.where(board == 1)[-1]


def sample(population, sample_size):
    random_indices = numpy.arange(len(population))
    numpy.random.shuffle(random_indices)
    return items(population, random_indices[:sample_size])


def item(values, index):
    return values[index]


def items(values, indices, count=-1, container_type=list):
    if isinstance(values, numpy.ndarray):
        try:
            return values[indices]
        except IndexError as _:
            return values[numpy.fromiter(indices, dtype='int', count=count)]
    return container_type(imap(values.__getitem__, indices))


def swap(board, indices):
    new_board = board.copy()
    new_board[indices[::-1]] = board[indices]  # swap
    return new_board


def identity(value):
    return value


def factorial(value, cache={}):
    if value not in cache:
        cache[value] = math.factorial(value)
    return cache[value]


def ifactoradic(perm):  # convert a permutation of indices to its factorial base as a generator ...
    perm = numpy.asarray(perm)
    return (digit - numpy.sum(perm[:index] < digit) for index, digit in enumerate(perm))


def permutation_lexi_order(perm):  # get the decimal lexicographical order of a giving permutation ...
    return sum(imap(numpy.multiply, ifactoradic(perm), imap(factorial, reversed(xrange(len(perm))))))


def idecimal_to_factoradic(dec_value, length):  # get the permutation of a giving decimal lexicographical order
    for factorial_base in imap(factorial, reversed(xrange(length))):
        digit, perm_order = divmod(dec_value, factorial_base)
        yield digit


def decimal_to_factoradic(dec_value, length):
    return tuple(idecimal_to_factoradic(dec_value, length))


def dec_lexi_order_to_ipermutation(dec_perm_order, length):
    # get the permutation as an iterator from its the decimal lexicographical order
    return imap(list(xrange(length)).pop, idecimal_to_factoradic(dec_perm_order, length))




