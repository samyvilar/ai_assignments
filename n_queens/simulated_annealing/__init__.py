__author__ = 'samy.vilar'
__date__ = '12/26/12'
__version__ = '0.0.1'

import sys
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')

from n_queens import calc_collision

import numpy
import random

def simulated_annealing(
        objective_function,
        initial_solution,
        permutation_function,
        max_number_of_iterations = 10**6,
        temperature = 1000,
        alpha = 0.99
        ):

    current_solution = initial_solution.copy()
    best_error = current_error = objective_function(current_solution)

    max_iteration = max_number_of_iterations
    while max_number_of_iterations and best_error: # while we can still do better
        new_solution = permutation_function(current_solution)
        new_error = objective_function(new_solution)

        if new_error < current_error:
            current_solution = new_solution
            current_error = new_error
            if new_error < best_error:
                best_error = new_error
                best_solution = new_solution
                print "new best solution found @iteration: {iteration} new error {new_error}".format(
                    iteration = max_iteration - max_number_of_iterations,
                    new_error = new_error
                )
            if not new_error:
                print "Found best solution @iteration: {iteration}".format(
                    iteration = max_iteration - max_number_of_iterations,
                )
        elif numpy.random.random() < numpy.e**(new_error - current_error)/temperature:
            current_solution = new_solution
            current_error = new_error

        temperature *= alpha
        max_number_of_iterations -= 1

    return best_solution

def solve_n_queens_problem(number_of_queens):

    if number_of_queens == 1:  # solution is trivial for n = 1
        return numpy.ones(1)
    if number_of_queens < 4:   # no solution exists for n = (2, 3)
        return None

    all_rows = numpy.arange(number_of_queens)
    random_row_indices = random.sample(all_rows, number_of_queens) # initial random rows.
    best_solution = numpy.zeros((number_of_queens, number_of_queens))
    best_solution[all_rows, random_row_indices] = 1 # initial solution

    def permutation_function(current_solution):
        rand_cols = random.sample(all_rows, 2) # find two random columns.
        new_solution = current_solution.copy()    # copy solution.
        new_solution[:, rand_cols[::-1]] = current_solution[:, rand_cols] # swap columns.
        return new_solution


    return simulated_annealing(
        calc_collision,
        best_solution,
        permutation_function,
    )

if __name__ == '__main__':
    solve_n_queens_problem(10)
