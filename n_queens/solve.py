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

import time
import argparse

import simulated_annealing
import genetic_algorithm
import genetic_programming

from utils import calc_collision, py_diagonal_collisions, new_initialized_board

packages = simulated_annealing, genetic_algorithm, genetic_programming


def parse_arguments():
    parse = argparse.ArgumentParser()
    parse.add_argument('--number_of_queens', default=10, nargs='?', type=int)
    parse.add_argument('--algorithms', default=packages, nargs='+', type=lambda name: globals()[name])
    parse.add_argument('--max_iterations', default=10000, nargs='?', type=int)
    parse.add_argument('--population_size', default=1000, nargs='?', type=int,
                       help='Size of the population to be used to contain genes.')

    args = parse.parse_args()
    return args.algorithms, args.number_of_queens, args.max_iterations, args.population_size


if __name__ == '__main__':
    packages, number_of_queens, max_iterations, population_size = parse_arguments()

    for package in packages:
        start = time.time()
        solution = package.solve_n_queens_problem(
            number_of_queens,
            max_iterations=max_iterations,
            population_size=population_size
        )
        end = time.time()
        if py_diagonal_collisions(solution):
            print('failed to locate solution increase the number of iterations!')

        print('algorithm: {algorithm} error: {error} time: {elapse_time}s number_of_queens: {number_of_queens}'.format(
            algorithm=package.__name__,
            error=py_diagonal_collisions(solution),
            elapse_time=end - start,
            number_of_queens=number_of_queens,
        ))
        print("closest solution: perm {0} \n{1}\n\n".format(solution, new_initialized_board(solution)))