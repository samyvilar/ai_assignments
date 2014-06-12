"""
Genetic Algorithm, is meta-heuristic search algorithm,
where the search process emulates natural evolution.

We begin with a population of n individuals, each randomly generated
from a uniform distribution.

At each iteration cycle:
    1) We select k individuals where k <= n, the selection may be done randomly
        or following a special algorithm, in this case fitness proportionate selection or roulette-wheel selection
    2) We apply genetic operators such as:
        - crossover
            swap portion(s) from each parents chromosome to create a new member.
        - mutation
            flip certain portions of a members chromosome.
    3) repeat step two until we have generated enough members to constitute a new population.
    4) check if new population has a member that meets our ending criteria,
        if not repeat until so, or computational resources exhausted.

Being that then n-queens program can be viewed as finding a permutation on how two place a queen in each row
where non of the queens are attacking each other, we can then view the population as a set of permutations randomly
sampled from the n! possible unique permutations.

Mutations occur with a probability of .2, and applied by simply swapping an element from the permutation,
    in essence swapping a row.

Crossbreeding as a bit trickier since each solution has a unique set of values and
merging two solutions would with high probability violate this property creating a solution with a lower fitness
(row column collisions would occur)
so instead will use the solution inversion sequence, which measures how much out order each element is, this has the
added benefit that duplicate values are allowed.
"""
__author__ = 'samy.vilar'

from itertools import imap, repeat, izip, chain, cycle, starmap, count, izip_longest
from bisect import bisect_left

import numpy
import hashlib
from numpy import cumsum, sum, divide
from numpy.random import random

from utils import calc_collision, diagonal_collisions, sample, items, item, ipermutation_inversion, permutation_inversion, permutation_from_inversion
from utils import new_initialized_board, swap, board_element_type


def genetic_algorithm(
    population,
    selection,
    genetic_operators,
    sort_population,
    error_func,
    sample_percentage=.35,
    max_iterations=10**6,
):
    population_size = len(population)
    max_iteration = max_iterations
    sample_size = int(sample_percentage * population_size)
    smallest_error = error_func(population[0])

    while max_iterations and smallest_error:
        population = genetic_operators(selection(population, sample_size), population_size)
        max_iterations -= 1
        # print 'smallest_error: ', smallest_error, 'curr smallest_error: ', error_func(population[0])

        if error_func(population[0]) < smallest_error:
            smallest_error = error_func(population[0])
            print("improved solution found @iteration: {iteration} new error {new_error}".format(
                iteration=max_iteration - max_iterations, new_error=smallest_error
            ))
    print('total iterations: {0}'.format(max_iteration - max_iterations))
    return population[0]


def fitness_function(solution, cache={}):
    hash_value = hashlib.sha1(solution).hexdigest()
    if hash_value not in cache:
        cache[hash_value] = len(solution) - diagonal_collisions(solution)
    return cache[hash_value]


def get_selection(fitness_function):
    def selection(population, sample_size, fitness_function=fitness_function):
        # fitness proportionate selection/roulette-wheel selection,
        # assumes that the population has being sorted from best to worse, high fitness to low fitness ...
        fitnesses = numpy.fromiter(imap(fitness_function, population), dtype='float', count=len(population))
        # normalized all the fitness values (collision) and take their accumulated sum to create a probability line
        prob_line = cumsum(divide(fitnesses, float(sum(fitnesses))))
        return items(population, imap(bisect_left, repeat(prob_line, sample_size), imap(apply, repeat(random))))
    return selection


selection = get_selection(fitness_function)


def get_sort_population(fitness_function):  # sort population based on fitness function,
    def sort_population(population, fitness_function=fitness_function):
        population.sort(key=fitness_function, reverse=True)
        return population
    return sort_population


sort_population = get_sort_population(fitness_function)


def solve_n_queens_problem(number_of_queens, population_size=10**3, max_iterations=10**4):
    assert 0 < number_of_queens < 256
    indices = numpy.arange(number_of_queens)

    # def swap_2_random_rows(population, all_rows=indices):
    #     perm, rand_rows = sample(population, 1)[0], sample(all_rows, 2)
    #     new_perm = perm.copy()
    #     new_perm[rand_rows[::-1]] = perm[rand_rows[0]], perm[rand_rows[1]]
    #     return new_perm
    swap_2_random_rows = lambda population, all_rows=indices: swap(sample(population, 1)[0], sample(all_rows, 2))

    numb_of_parents = 2
    chromo_length = number_of_queens/numb_of_parents
    slices = tuple(imap(
        apply,
        repeat(slice),
        izip_longest(*imap(xrange, (0, chromo_length), repeat(number_of_queens - 1), repeat(chromo_length))),
    ))

    def merge_2_random_solutions(population):
        return permutation_from_inversion(  # merge two solutions by merging their inversion sequence ...
            numpy.fromiter(
                chain.from_iterable(
                    imap(  # get inversion sequence from each donor parent ...
                        item,
                        imap(tuple, imap(permutation_inversion, sample(population, numb_of_parents))),
                        slices
                    )
                ),
                count=number_of_queens,
                dtype=board_element_type
            )
        )

    operators = merge_2_random_solutions, swap_2_random_rows

    def genetic_operators(population, sample_size, prob_of_mutation=.3):
        return sorted(
            imap(apply, imap(operators.__getitem__, random(sample_size) < prob_of_mutation), repeat((population,))),
            key=fitness_function,
            reverse=True
        )

    return genetic_algorithm(
        sorted(
            starmap(
                sample,
                repeat((numpy.arange(number_of_queens, dtype=numpy.uint8), number_of_queens), population_size)
            ),
            key=fitness_function,
            reverse=True
        ),
        selection,
        genetic_operators,
        sort_population,
        lambda perm: len(perm) - fitness_function(perm),
        max_iterations=max_iterations
    )

from itertools import permutations


def solve(n):
    cols = range(n)
    return [
        vec for vec in permutations(cols) if n == len(set(vec[i]+i for i in cols)) == len(set(vec[i]-i for i in cols))
    ]