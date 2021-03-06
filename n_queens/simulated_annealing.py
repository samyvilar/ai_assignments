"""
Simulated Annealing, is a meta-heuristic search algorithm, where the search process emulates annealing on metals.

Well we could search for a 'best' solution, by doing the following:
1) Initially choose a current solution (randomly) and assume its the 'best' solution.
2) If the best solution has being found either
because we have exhausted our resources or
indeed it is our 'best' solution, just return it.
3) If not, choose a new current solution,
by simply selecting it at random around our current solution,
and assume its 'better'.
4) If indeed it is better then take it.
5) If not take it anyway with a certain probability.
6) repeat.

One might ask, why are you jumping to a worse solution,
even if it is only with a certain probability?
Informally: To put it simply "we do this to avoid getting stuck in a local minimum".

Formally: we define a function f, called the 'objective function' that measures the
quality of each solution, where f(solution) >= 0 for all solutions,
and f(solution) == 0 where solution is our 'best solution'
and f(solution) > f(better_solution)
where better solution is closer to our best or ideal solution then solution.

In essence this function creates a surface, we are simply minimizing this function,
if we only minimize, ie we only take values that increase our objective,
we may get stuck in a local minimum,
since we can't really assume anything about this function or surface,
like it has only one minimum, the global minimum, or all local_minimums == global_minimums,
we have to assume it has many local_minimums that may not equal our global_minimum,
as such we have to leave a local minimum, for another local minimum,
until we hit the local_minimum == global_minimum, or computational resources exhausted.


We can represent a solution as a sequence of n numbers each from 0 to n - 1.
where the ith value represents the column position of a queen in the ith row.

We start by randomly placing a queen in each row,
define our objective function as the number of collisions of a giving solution,
note we only need the diagonal collisions since each column/row index pair is unique ...

Assume this is our best solution, if not look for a better solution, around this solution.
We do this by simply swapping a column at random.

Repeat until solution found.
"""

__author__ = 'samyvilar'

import numpy

from utils import diagonal_collisions, new_random_board, sample, swap, py_diagonal_collisions


def simulated_annealing(
    objective_function,
    initial_solution,
    permutation_function,
    max_iterations=10**6,
    temperature=1000,
    alpha=0.99
):

    best_solution = current_solution = initial_solution.copy()
    best_error = current_error = objective_function(current_solution)
    max_iteration = max_iterations

    while max_iterations and best_error:  # while we can still do better
        new_solution = permutation_function(current_solution)
        new_error = objective_function(new_solution)

        if new_error < current_error:
            current_solution, current_error = new_solution, new_error
            if new_error < best_error:
                best_error, best_solution = new_error, new_solution
                print("improved solution found @iteration: {iteration} new error {new_error}".format(
                    iteration=max_iteration - max_iterations, new_error=new_error
                ))
        elif numpy.random.random() < numpy.e**(new_error - current_error)/temperature:
            current_solution, current_error = new_solution, new_error

        temperature *= alpha  # anneal ...
        max_iterations -= 1

    if not best_error:
        print("Found solution @iteration: {0}".format(max_iteration - max_iterations))

    return best_solution


def solve_n_queens_problem(number_of_queens, max_iterations=10000, population_size=1000):
    assert number_of_queens >= 0
    if not number_of_queens:        # solution is trivial for n in {0, 1}
        return ()
    elif number_of_queens == 1:     # solution is trivial for n = 1
        return 1,
    elif number_of_queens < 4:      # no solution exists for n in {2, 3}
        return ()

    def permutation_function(current_solution, row_indices=numpy.arange(number_of_queens)):
        return swap(current_solution, sample(row_indices, 2))   # swap two random rows.

    return simulated_annealing(
        diagonal_collisions,
        new_random_board(number_of_queens),
        permutation_function,
        max_iterations=max_iterations,
        temperature=population_size
    )

