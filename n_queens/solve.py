#! /usr/bin/env python
"""
The n-queens problem, can be described as follows, giving an n-by-n chess board,
place n queens such as no queen may attack any other queen.

No algorithm exists, to solve this problem, as such we are left to simply search
through all the possible solutions.

The problem is that the solution is space is quite large!; formally its (n**2 choose n)
for example a 10-by-10 board may have 100!/10!90! or 17,310,309,456,440 possible solutions,
that's 17 trillion, so the naive 'brute force' approach is simply infeasible,
as n increases.

One way to shrink the solution space is to note, that we can only place a single queen
in each row, with a unique column as such our space shrinks to n! so for a 10-by-10
board we have 3,628,800 possible solutions, while n! grows much much slower than
(n**2 choose n), it still grows quite fast, and for sufficiently large n iterating
through this solution space is simple not possible.

With that in mind, here are three meta-heuristic search methods to solve this problem:
simulated_annealing
genetic_algorithm
genetic_programming

CODE REQUIREMENTS:
python >= 2.7
numpy
matplotlib
"""
__author__ = 'samyvilar'

import sys
import time

import simulated_annealing
import genetic_algorithm
import genetic_programming

from utils import calc_collision

if __name__ == '__main__':
    packages = simulated_annealing, genetic_algorithm, genetic_programming
    number_of_queens = (len(sys.argv) > 1 and int(sys.argv[1])) or 10

    for package in packages:
        start = time.time()
        solution = package.solve_n_queens_problem(number_of_queens)
        end = time.time()
        print('algorithm: {algorithm} error: {error} time: {elapse_time}s number_of_queens: {number_of_queens}'.format(
            algorithm=package.__name__,
            error=calc_collision(solution),
            elapse_time=end - start,
            number_of_queens=number_of_queens,
        ))
        print "solution:"
        print solution
        print "\n\n"