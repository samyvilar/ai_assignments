"""
Genetic Programming, is a meta-heuristic search algorithm, inspired by biological evolution,
that searches for computer programs, that can solve a defined task.

This is somewhat similar to genetic algorithms, and in fact it is!; buts its considered
a 'specialization' of genetic algorithms, with the exception being that
each member now is a computer program.

The same rules still apply, population -> selection -> crossbreeding -> repeat ...

This time around we are not looking for a computer program, where
a computer program can be defined as a set of operator and operands, usually visualized
as a tree.

But for the n-queens problem we really don't need to use a tree,
basically we start with a random solution and we look for a sequence of column-swaps
that would transform our initial solution to an optimal solution, using the least
number of swaps.

Note that we can slightly modify our solution so each program would have its own
initial random solution, this way they are all decoupled from each other and it
may be easier to run the program concurrently.
"""
__author__ = 'samy.vilar'

from itertools import izip, repeat, imap, starmap

import numpy
import random

from genetic_algorithm import genetic_algorithm, selection, get_sort_population
from utils import calc_collision, new_random_board, sample_population


def get_program_generator(max_depth, number_of_queens):
    def get_new_program(max_number_of_swaps=max_depth, row_indices=numpy.arange(number_of_queens)):
        def swap(columns, board):
            new_board = board.copy()
            new_board[:, columns[::-1]] = board[:, columns]  # swap columns ...
            return new_board
        # program (chromosome ... ) is nothing more then sequence of random swaps ...
        return list(
            izip(
                repeat(swap, numpy.random.randint(1, max_number_of_swaps)),
                starmap(random.sample, repeat((row_indices, 2)))
            )
        )
    return get_new_program


def cross_over(population):
    parents = sample_population(population, 2)
    return parents[0][:random.randint(0, len(parents[0]) - 1)] + parents[1][random.randint(0, len(parents[1]) - 1):]


def hoist_mutation(population):
    program = sample_population(population, 1)
    return program[random.randint(0, len(program) - 1):]


def shrink_mutation(population):
    program = sample_population(population, 1)
    new_program = list(program)
    del new_program[random.randint(1, len(program)):]
    return new_program


def get_genetic_operators(number_of_queens, get_new_program):
    def subtree_mutation(population):
        return cross_over((sample_population(population, 1), get_new_program()))

    def constant_mutation(population):
        program = sample_population(population, 1)
        rand_index = random.randint(0, len(program) - 1)
        new_program = list(program)
        new_program[rand_index][1][:] = \
            [(program[rand_index][1][0] + random.randint(0, number_of_queens)) % number_of_queens,
             (program[rand_index][1][1] + random.randint(0, number_of_queens)) % number_of_queens]
        return new_program

    rules = subtree_mutation, constant_mutation, shrink_mutation, hoist_mutation

    def genetic_operators(population, sample_size):
        # Each solution is unique any composite solution(s) would just add collisions to row/column ...
        prob_of_reproduction = .025
        prob_of_mutation = .15

        def random_value_to_func(rand_value):
            if rand_value <= prob_of_reproduction:
                return lambda population: sample_population(population, 1)
            elif rand_value <= prob_of_mutation:
                return rules[random.randint(0, len(rules) - 1)]
            else:
                return cross_over
        return map(
            apply,
            imap(random_value_to_func, imap(apply, repeat(numpy.random.random, sample_size))),
            repeat((population,))
        )


def get_fitness_function(initial_board):
    def fitness_function(program, new_board=initial_board):
        collision = None
        for index, prog in enumerate(program):
            new_board = prog[0](prog[1], new_board)
            collision = calc_collision(new_board)
            if not collision:
                del program[index + 1:]
                break
        return new_board.size - collision
    return fitness_function


def solve_n_queens_problem(number_of_queens, population_size=10**3, max_depth=10**2, max_number_iterations=10**6):
    board = new_random_board(number_of_queens)
    fitness_function = get_fitness_function(board)
    sort_population = get_sort_population(fitness_function)

    def error_func(program):
        return board.size - fitness_function(program)

    best_program = genetic_algorithm(
        sort_population(map(apply, repeat(get_program_generator(max_depth, number_of_queens), population_size))),
        selection,
        get_genetic_operators(number_of_queens, get_program_generator(max_depth, number_of_queens)),
        sort_population,
        error_func,
        max_number_of_iterations=max_number_iterations,
        sample_size=.35,
    )

    return reduce(lambda board, program: program[0](program[1], board), best_program, board)