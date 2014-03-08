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

For the n-queens problem, we take our n-by-n matrix of 0 and 1s and simply generate
a population of possible solutions, for each possible solution measure
the normalize fitness value (collisions) as prescribed by the roulette-wheel selection
algorithm, we select a subset of the population for crossbreeding and mutations.

Mutations occur with a probability of .1, while crossbreeding is at .9.
Mutations are implemented as a column swap very similar to simulated annealing.

Cross breeding is a bit trickier, the main issue is that each chromosome has to be unique,
if we arbitrary take chromosomes from each parent than we may loose this uniqueness.
Instead we take a chromosome from either parent with a probability .5, but if this
chromosome is already present check the other parent, if also present, then
just select one at random from the available set.
"""
__author__ = 'samy.vilar'
from itertools import imap, repeat, izip, chain
from bisect import bisect_left

import numpy
from numpy import cumsum, sum, divide
from numpy.random import random

from utils import calc_collision, irandom_board, sample, new_initialized_board, items, swap, board_element_type


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
        population = sort_population(genetic_operators(selection(population, sample_size), population_size))
        max_iterations -= 1
        if error_func(population[0]) < smallest_error:
            smallest_error = error_func(population[0])
            print("improved solution found @iteration: {iteration} new error {new_error}".format(
                iteration=max_iteration - max_iterations, new_error=smallest_error
            ))
    print('total iterations: {0}'.format(max_iteration - max_iterations))
    return population[0]


def fitness_function(solution):
    return solution.size - calc_collision(solution)


def get_selection(fitness_function):
    def selection(population, sample_size, fitness_function=fitness_function):
        # fitness proportionate selection/roulette-wheel selection,
        # assumes that the population has being sorted from best to worse, high fitness to low fitness ...
        fitnesses = numpy.fromiter(imap(fitness_function, population), dtype='float', count=len(population))
        # normalized all the fitness values (collision) and take their accumulated sum to create a probability line
        probability_l = cumsum(divide(fitnesses, float(sum(fitnesses))))  # probability line ...
        return items(population, imap(bisect_left, repeat(probability_l, sample_size), imap(apply, repeat(random))))
    return selection


selection = get_selection(fitness_function)


def get_sort_population(fitness_function):  # sort population based on fitness function,
    def sort_population(population, fitness_function=fitness_function):
        population_fitness = numpy.fromiter(imap(fitness_function, population), dtype='float', count=len(population))
        sum_of_errors = float(population_fitness.sum())
        if isinstance(population, numpy.ndarray):
        # sort the population based on their relative fitness in desc order, from best to worse ...
            return population[numpy.argsort(population_fitness / sum_of_errors)][::-1]
        population.sort(key=lambda population: -fitness_function(population) / sum_of_errors)  # negate to sort in desc
        return population
    return sort_population


sort_population = get_sort_population(fitness_function)


def solve_n_queens_problem(number_of_queens, population_size=10**3, max_iterations=10**4):
    indices = numpy.arange(number_of_queens)

    def swap_2_random_columns(population, all_rows=indices):
        return swap(sample(population, 1)[0], sample(all_rows, 2))

    def merge_2_random_solutions(population, possible_columns=indices):
        parents = sample(population, 2)  # select parents at random ...
        columns = numpy.where(parents == 1)[-1].reshape(2, number_of_queens)
        random_indices = numpy.fromiter(  # randomly select a chromosomes from each parent at random ...
            imap(columns.__getitem__, izip(random(number_of_queens) < 0.5, possible_columns)), 'int', number_of_queens
        )
        missing_elems = numpy.setdiff1d(possible_columns, random_indices)  # check if we have any missing ...
        if len(missing_elems):  # missing chromosomes, solutions has row/column collisions ...
            numpy.random.shuffle(missing_elems)  # shuffle the missing rows since setdiff1d sorts them ...
            random_indices[numpy.setdiff1d(possible_columns, numpy.unique(random_indices, True)[1])] = missing_elems
        return new_initialized_board(random_indices)  # create a new board from randomly selected entries from parents.

    operators = merge_2_random_solutions, swap_2_random_columns

    def genetic_operators(population, sample_size, prob_of_mutation=.3):
        new_population_shape = sample_size, number_of_queens, number_of_queens
        return numpy.fromiter(
            chain.from_iterable(chain.from_iterable(
                imap(apply, imap(operators.__getitem__, random(sample_size) < prob_of_mutation), repeat((population,)))
            )),
            dtype=population.dtype,
            count=numpy.product(new_population_shape)
        ).reshape(new_population_shape)

    population_shape = population_size, number_of_queens, number_of_queens
    return genetic_algorithm(
        sort_population(
            numpy.fromiter(
                chain.from_iterable(imap(irandom_board, repeat(number_of_queens, population_size))),
                dtype=board_element_type,
                count=numpy.product(population_shape)
            ).reshape(population_shape)
        ),
        selection,
        genetic_operators,
        sort_population,
        calc_collision,
        max_iterations=max_iterations
    )
