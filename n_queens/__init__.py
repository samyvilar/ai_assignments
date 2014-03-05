__author__ = 'samy.vilar'
__date__ = '12/26/12'
__version__ = '0.0.1'

from collections import defaultdict

import numpy
import sys
import hashlib


def diagonals(a):
    rows, cols = a.shape
    if cols > rows:
        a = a.T
        rows, cols = a.shape
    fill = numpy.zeros(((cols - 1), cols), dtype = a.dtype)
    stacked = numpy.vstack((a, fill, a))
    major_stride, minor_stride = stacked.strides
    strides = major_stride, minor_stride * (cols + 1)
    shape = (rows + cols - 1, cols)
    return numpy.lib.stride_tricks.as_strided(stacked, shape, strides)


# make sure to properly set the cache size, depending on the the system
# default is 3 gigs.
def calc_collision(solution, cache=defaultdict(defaultdict), cache_size_bytes=3 * 10**9):
    column_axis, row_axis = 0, 1
    hash = hashlib.sha1(solution).hexdigest()

    if hash in cache:
        # assert not numpy.sum(cache[hash]['solution'] - solution) # TODO: check for collisions.
        return cache[hash]['diagonal_collision']

    #since we are shuffling the columns their will never be any collision around the rows or columns.
    #queens_in_rows = solution.sum(axis = row_axis)
    #row_collision = queens_in_rows[numpy.where(queens_in_rows > 1)].sum()
    #queens_in_columns = solution.sum(axis = column_axis)
    #column_collision = queens_in_columns[numpy.where(queens_in_columns > 1)].sum()

    left_diagonal_queens = diagonals(solution).sum(axis = row_axis)
    right_diagonal_queens = diagonals(numpy.fliplr(solution)).sum(axis = row_axis)

    diagonal_collision = left_diagonal_queens[numpy.where(left_diagonal_queens > 1)].sum() +\
                         right_diagonal_queens[numpy.where(right_diagonal_queens > 1)].sum()

    #if sys.getsizeof(cache) < cache_size_bytes:
    cache[hash]['diagonal_collision'] = diagonal_collision
    #cache[hash]['solution'] = solution # TODO: check for collisions.

    return diagonal_collision

if __name__ == '__main__':
    import simulated_annea
    import genetic_algor
    import genetic_programming
    import time

    packages = [simulated_annea, genetic_algor, genetic_programming]
    number_of_queens = 10

    for package in packages:
        start = time.time()
        solution = package.solve_n_queens_problem(number_of_queens)
        end = time.time()
        print "algorithm: {algorithm} error: {error} time: {elapse_time}s number_of_queens: {number_of_queens}".format(
            algorithm = package.__name__,
            error = calc_collision(solution),
            elapse_time = end - start,
            number_of_queens = number_of_queens,
        )
        print "solution:"
        print solution
        print "\n\n"