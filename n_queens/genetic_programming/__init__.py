__author__ = 'samy.vilar'
__date__ = '12/26/12'
__version__ = '0.0.1'

import numpy
import random

from n_queens import calc_collision
from n_queens.genetic_algorithm import genetic_algorithm

def solve_n_queens_problem(
        number_of_queens,
        population_size = 10**3,
        max_depth = 10**2,
        max_number_iterations = 10**6,
    ):
    board = numpy.zeros((number_of_queens, number_of_queens))
    board_size_indices = numpy.arange(number_of_queens)
    board[
        board_size_indices,
        random.sample(board_size_indices, number_of_queens)
    ] = 1

    def get_new_program(max_number_of_swaps = max_depth):
        def swap(columns, board):
            columns = columns
            new_board = board.copy()
            new_board[:, columns[::-1]] = board[:, columns]
            return new_board

        return [[swap, random.sample(board_size_indices, 2)] # program is nothing more then sequence of swaps.
            for index in xrange(max_number_of_swaps)
        ]


    def fitness_function(program):
        new_board = board
        for index, pro in enumerate(program):
            new_board = pro[0](pro[1], new_board)
            collision = calc_collision(new_board)
            if not collision: # if there where no collisions remove any remaining operations and just return.
                del program[index + 1:]
                break
        return collision

    def sort_population(population):
        all_fitness = numpy.asarray(map(fitness_function, population))
        sum_of_errors = sum(all_fitness)
        return sorted(
            population,
            key = lambda program: fitness_function(program)/sum_of_errors,
            reverse = True,
        )


    def selection(population, count): # fitness proportionate selection or roulette-wheel selection
        all_fitness = numpy.asarray(map(fitness_function, population))
        sum_of_errors = all_fitness.sum()
        normalized_acc_fitness_values = numpy.add.accumulate(all_fitness / sum_of_errors)
        new_population = [
            population[
                numpy.where(
                    normalized_acc_fitness_values > numpy.random.random())[0][0]
                ]   for index in xrange(int(count))
            ]
        return new_population


    def genetic_operators(population, size):
        # Each solution is unique any composite solution(s) would just
        # add collisions to row/column ...
        new_population = []
        prob_of_reproduction = .025
        prob_of_mutation = .15

        def cross_over(parents):
            parent_0, parent_1 = parents
            parent_0_index = random.randint(0, len(parent_0) - 1)
            parent_1_index = random.randint(0, len(parent_1) - 1)
            return parent_0[:parent_0_index] + parent_1[parent_1_index:]

        new_population = []
        for index in xrange(size):
            rand_number = numpy.random.random()
            if rand_number <= prob_of_reproduction:
                new_program, = random.sample(population, 1)
            elif rand_number <= prob_of_mutation:
                new_program, = random.sample(population, 1)
                def subtree_mutation(program):
                    new_program = get_new_program()
                    return cross_over((program, new_program))
                def constant_mutation(program):
                    rand_index = random.randint(0, len(program) - 1)
                    new_program = list(program)
                    new_program[rand_index][1][:] = \
                        [(program[rand_index][1][0] + random.randint(0, number_of_queens)) % number_of_queens, \
                         (program[rand_index][1][1] + random.randint(0, number_of_queens)) % number_of_queens]
                    return new_program
                def shrink_mutation(program):
                    new_program = list(program)
                    del new_program[random.randint(1, len(program)):]
                    return new_program
                def hoist_mutation(program):
                    return program[random.randint(0, len(program) - 1):]
                rule = [subtree_mutation, constant_mutation, shrink_mutation, hoist_mutation]
                new_program = rule[random.randint(0, len(rule) - 1)](new_program)
            else:
                new_program = cross_over(random.sample(population, 2))

            new_population.append(new_program)
        return new_population

    population = [
        get_new_program()
            for index in xrange(population_size)
    ]



    program = genetic_algorithm(
        sort_population(population),
        selection,
        genetic_operators,
        sort_population,
        fitness_function,
        max_number_of_iterations = max_number_iterations,
        sample_size = .35,
    )

    new_board = board
    for index, pro in enumerate(program):
        new_board = pro[0](pro[1], new_board)
    return new_board



if __name__ == '__main__':
    from n_queens.genetic_programming import *
    solve_n_queens_problem(20)







 
