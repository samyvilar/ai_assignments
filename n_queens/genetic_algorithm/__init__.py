__author__ = 'samy.vilar'
__date__ = '12/26/12'
__version__ = '0.0.1'

import numpy
import random
from n_queens import calc_collision

import gc

default_max_number_of_iterations = 10**6

def genetic_algorithm(
        population,
        selection,
        genetic_operators,
        sort_population,
        fitness_function,
        sample_size = .35,
        max_number_of_iterations = default_max_number_of_iterations,
        ):
    population_size = len(population)


    max_iteration = max_number_of_iterations
    while max_number_of_iterations and fitness_function(population[-1]):
        population = sort_population(
            genetic_operators(
                selection(population, sample_size * population_size),
                population_size,
            )
        )

        if not fitness_function(population[-1]):
            print "best solution found @iteration: {iteration}".format(
                iteration = max_iteration - max_number_of_iterations,
            )
            break
        max_number_of_iterations -= 1
        _ = gc.collect()



    print "best solution found after {iteration} iterations, fitness: {fitness}".format(
        iteration = max_iteration - max_number_of_iterations,
        fitness = fitness_function(population[-1]),
    )

    return population[-1]


def solve_n_queens_problem(
        number_of_queens,
        population_size = 10**3,
    ):

    population = numpy.zeros((population_size, number_of_queens, number_of_queens))
    random_solutions = numpy.asarray([xrange(number_of_queens)] * population_size)
    _ = map(numpy.random.shuffle, random_solutions)

    all_rows = numpy.arange(number_of_queens)
    for index, sol in enumerate(population):
        sol[all_rows, random_solutions[index]] = 1

    def selection(population, count): # fitness proportionate selection or roulette-wheel selection
        all_fitness = numpy.asarray(map(calc_collision, population))
        sum_of_errors = all_fitness.sum()
        normalized_acc_fitness_values = numpy.add.accumulate(all_fitness / sum_of_errors)
        new_population = [
                population[
                            numpy.where(
                                normalized_acc_fitness_values > numpy.random.random())[0][0]
                          ]
                    for index in xrange(int(count))
        ]
        return numpy.asarray(new_population)

    def genetic_operators(population, size):
        # Each solution is unique any composite solution(s) would just
        # add collisions to row/column ...
        new_population = []
        prob_of_mutation = .1
        for index in xrange(size):
            rand_prob = numpy.random.random()
            if rand_prob < prob_of_mutation:
                solution, = random.sample(population, 1)
                rand_cols = random.sample(all_rows, 2) # find two random columns.
                new_solution = solution.copy()
                new_solution[:, rand_cols[::-1]] = solution[:, rand_cols] # swap columns.
                new_population.append(new_solution)
            else:
                parents = random.sample(population, 2)
                new_solution = numpy.zeros((number_of_queens, number_of_queens))
                set_columns = {}
                possible_columns = set(xrange(number_of_queens))
                for row_index in xrange(number_of_queens):
                    parent = parents[::] if numpy.random.random() < .5 else parents[::-1]
                    col_index_0, col_index_1 = \
                            numpy.where(parent[0][row_index] == 1)[0][0], \
                            numpy.where(parent[1][row_index] == 1)[0][0]


                    if col_index_0 not in set_columns:
                        col_index = col_index_0
                    elif col_index_1 not in set_columns:
                        col_index = col_index_1
                    else:
                        col_index = random.sample(possible_columns - set(set_columns), 1)[0]

                    new_solution[row_index, col_index] = 1
                    set_columns[col_index] = col_index
            new_population.append(new_solution)
        return numpy.asarray(new_population)



    def sort_population(population):
        all_fitness = numpy.asarray(map(calc_collision, population))
        sum_of_errors = all_fitness.sum()
        return numpy.asarray(
            sorted(
                population,
                key = lambda solution: calc_collision(solution)/sum_of_errors,
                reverse = True,
            )) # sort in descending order.

    return genetic_algorithm(
        sort_population(population),
        selection,
        genetic_operators,
        sort_population,
        calc_collision
    )



if __name__ == '__main__':
    solve_n_queens_problem(10)





 
