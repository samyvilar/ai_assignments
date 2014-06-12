"""
Genetic Programming, is a meta-heuristic search algorithm, inspired by biological evolution,
that searches for computer programs, that can solve a defined task.

This is somewhat similar to genetic algorithms, and in fact it is!; buts its considered
a 'specialization' of genetic algorithms, with the exception being that
each member now is a computer program.

The same rules still apply, population -> selection -> crossbreeding -> repeat ...

This time around we are looking for a computer program, where
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

from itertools import izip, repeat, imap, starmap, cycle, takewhile

import numpy
from numpy.random import randint, random

from genetic_algorithm import genetic_algorithm, get_selection, get_sort_population
from utils import calc_collision, new_random_board, sample, identity, swap, item


def program_generator(max_depth, number_of_queens, initial_board):
    def new_program(max_number_of_swaps=max_depth, row_indices=numpy.arange(number_of_queens)):
        class Program(object):  # program (chromosome ... ) is nothing more then sequence of random swaps ...
            def __init__(self, swaps):
                self.initial_board = initial_board
                # self.initial_board = new_random_board(number_of_queens)
                self.swaps = tuple(swaps)

            def __len__(self):
                return len(self.swaps)

            @property
            def collisions(self):
                if not hasattr(self, '_collisions'):
                    board = self.initial_board
                    collision = calc_collision(board)
                    index = -1
                    for index, columns in enumerate(takewhile(lambda _: collision, self.swaps)):
                        board = swap(board, columns)
                        collision = calc_collision(board)
                        if not collision:
                            break
                    if not collision:
                        self.swaps = self.swaps[:index + 1]
                    self._collisions = collision
                return self._collisions
                    # self.collisions = calc_collision(reduce(swap, self.swaps, self.initial_board))

            def __call__(self):  # fitness function ...
                return number_of_queens - self.collisions

            def __getitem__(self, item):  # if slicing return a new program otherwise return swap value ...
                return ((isinstance(item, slice) and Program) or identity)(self.swaps[item])

            def __add__(self, other):
                return Program(self.swaps + other.swaps)

        return Program(starmap(sample, repeat((row_indices, 2), randint(1, max_number_of_swaps + 1))))
    return new_program


def cross_over(population):  # select 2 random programs create a new program by randomly joining two sequences ..
    parents = sample(population, 2)
    return reduce(
        type(parents[0]).__add__,
        imap(
            item,
            parents,
            starmap(
                slice,
                izip(
                    imap(randint, imap(len, parents)),
                    imap(apply, cycle((lambda _: None, len)), imap(tuple, imap(repeat, parents, repeat(1))))
                )
            )
        )
    )


def hoist_mutation(population):  # randomly select a program, create a new program using a randomly selected subtree
    program = sample(population, 1)[0]
    return program[randint(len(program)):]


def get_shrink_mutation(number_of_queens, get_new_program):
    def shrink_mutation(population):  # replace subtree with a new terminal ...
        program = sample(population, 1)[0]
        return program[randint(len(program)):] + get_new_program(1)
    return shrink_mutation


def get_subtree_mutation(get_new_program):  # replace entire subtree with new program ...
    def subtree_mutation(population):
        program = sample(population, 1)[0]
        return program[:randint(len(program))] + get_new_program()
    return subtree_mutation


def get_constant_mutation(number_of_queens):
    def constant_mutation(population):
        program = sample(population, 1)[0]  # randomly select a program
        node = program[randint(len(program))]  # random select a node (swap command) get param
        random_index = randint(len(node))   # select indices at random
        node[randint(len(node))] = randint(number_of_queens)  # update the swap command ..
        return program
    return constant_mutation


def get_genetic_operators(number_of_queens, get_new_program):
    rules = get_subtree_mutation(get_new_program), \
        get_constant_mutation(number_of_queens), \
        get_shrink_mutation(number_of_queens, get_new_program), \
        hoist_mutation

    def genetic_operators(population, sample_size):
        prob_of_reproduction, prob_of_mutation = .025, .15

        def get_oper(rand_value):
            if rand_value <= prob_of_reproduction:  # entry doesn't reproduce ...
                return lambda population: sample(population, 1)[0]
            elif rand_value <= prob_of_mutation:
                return rules[randint(len(rules))]
            else:
                return cross_over  # probability of crossover .75 ...
        return list(imap(apply, imap(get_oper, imap(apply, repeat(random, sample_size))), repeat((population,))))
    return genetic_operators


def solve_n_queens_problem(number_of_queens, population_size=10**3, max_depth=200, max_iterations=10**3):
    board = new_random_board(number_of_queens)

    fitness_function = apply
    sort_population = get_sort_population(fitness_function)
    create_new_program = program_generator(max_depth, number_of_queens, board)

    def error_func(program):
        return program.collisions

    best_program = genetic_algorithm(
        sort_population(map(apply, repeat(create_new_program, population_size))),
        get_selection(fitness_function),
        get_genetic_operators(number_of_queens, create_new_program),
        sort_population,
        error_func,
        max_iterations=max_iterations,
        sample_percentage=.35,
    )

    return reduce(swap, best_program.swaps, best_program.initial_board)