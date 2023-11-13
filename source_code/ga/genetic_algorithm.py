import random
# from numpy import idxmax

from data_preperation.data_preprocessing import DataPreprocessing
# import pandas as pd
import numpy as np


class GeneticAlgorithm:
    def __init__(self, data: DataPreprocessing, initial_population_size: int = 100, max_iter: int = 100) -> None:
        self.population_size: int = initial_population_size
        self.data: DataPreprocessing = data
        self.max_iter = max_iter
        # self.population: list[list] = []
        self.population: dict = {}
        # self.chromosome_length = self.getChromosomeLength()
        self.evaluation_results: dict = {}
        self.first_free_idx = 0
        # self.generateInitialPopulation()
        # self.evaluationFunction(1)

        # print(self.population)

    '''
    @brief: generates initial population using chosen method:
    - version 1: random elements in previously calculated ranges
    - version 2: random element from the given database is chosen
    '''
    def generateInitialPopulation(self, pop_type: int = 1) -> bool:
        if pop_type == 1:
            # version 1 - create population with random elements in previously calculated ranges
            for idx in range(0, self.population_size):
                chromosome = []
                for label, vals in self.data.min_max_column_vals.items():
                    if vals[2] == 'float64':
                        chromosome.append(random.uniform(vals[0], vals[1]))
                    else:
                        chromosome.append(random.randint(vals[0], vals[1]))
                self.population[idx] = chromosome
                # print(chromosome)
        else:
            # version 2 - take random elements from the given dataframe
            for idx in range(0, self.population_size):
                chromosome = []
                num = random.randint(0, self.data.processed_dataframe.shape[0] - 1)
                for col_name, val in self.data.processed_dataframe.iloc[num].items():
                    if col_name not in ["id", "label"]:
                        chromosome.append(val)
                self.population[idx] = chromosome
        self.first_free_idx = self.population_size
        print(len(self.population[0]))
        return True

    '''
    @brief: evaluation function for given element od the population is being calculated 
    '''
    def evaluationFunction(self, idx, weights) -> float:
        # version 1
        # result = 0
        # indexes_to_check = [i for i in range(0, len(self.population[idx]))]
        # for i in self.data.attack_df.itertuples():
        #     for column in indexes_to_check:
        #         if i[column + 2] == self.population[idx][column]:
        #             # print(self.data.column_labels[column])
        #             result += weights[column]
        #             indexes_to_check.remove(column)
        # # TODO: LEARN WHAT THIS RANKING SHOULD BE
        # ranking = 1  # temporary solution
        # result = 1 - (result * ranking) / 100
        #
        # print(result)
        # return result

        # version 2
        result = 0
        idx_appears = []
        quantity = [0 for _ in range(self.getChromosomeLength())]

        for i in self.data.attack_df.itertuples():
            for column in range(0, self.getChromosomeLength()):
                if i[column + 2] == self.population[idx][column]:
                    if column not in idx_appears:
                        idx_appears.append(column)
                    # print(self.data.column_labels[column])
                    quantity[column] += 1
                    # result += weights[column]
                    # indexes_to_check.remove(column)

            # print(i[])
        result = len(idx_appears)
        # TODO: LEARN WHAT THIS RANKING SHOULD BE
        ranking = 1  # temporary solution
        x = 0
        for i in quantity:
            x += i
        # x += [i for i in quantity]
        result = x - (result * ranking) / 100
        print(quantity.index(max(quantity)))
        # print(result)
        return result

    def generateWeights(self, rand: bool = False) -> list:
        weights = []
        if rand:
            for i in range(0, self.getChromosomeLength()):
                weights.append(random.randint(1, 15))
        return weights

    def mutation(self, idx: int):
        elem_to_be_mutated = random.randint(0, len(self.population[idx]) - 1)
        col_name = self.data.column_labels[elem_to_be_mutated]
        if self.data.min_max_column_vals[col_name][-1] == "float64":
            self.population[idx][elem_to_be_mutated] = random.uniform(self.data.min_max_column_vals[col_name][0],
                                                                      self.data.min_max_column_vals[col_name][1])
        else:
            self.population[idx][elem_to_be_mutated] = random.randint(self.data.min_max_column_vals[col_name][0],
                                                                      self.data.min_max_column_vals[col_name][1])

        return True

    def selection(self):

        return True

    def crossing(self, id1: int, id2: int) -> bool:
        point_of_crossing = random.randint(1, len(self.population[id1]))
        child1 = [self.population[id1][i] for i in range(0, point_of_crossing)]
        for i in range(point_of_crossing, len(self.population[id2])):
            child1.append(self.population[id2])
        child2 = [self.population[id2][i] for i in range(0, point_of_crossing)]
        for i in range(point_of_crossing, len(self.population[id2])):
            child2.append(self.population[id1])

        self.population[self.first_free_idx] = child1
        self.population[self.first_free_idx + 1] = child2
        self.first_free_idx += 2
        return True

    def getChromosomeLength(self) -> int:
        result = len(self.data.column_labels) - 2
        # print(result)
        # return len(self.population[])
        return result

    def geneticAlgotrithmLoop(self):
        # generate initial population
        self.generateInitialPopulation()
        # evaluate initial population
        for keys, _ in self.population.items():
            self.evaluation_results[keys] = self.evaluationFunction(keys, self.generateWeights(True))
        # make selection
        self.selection()
        # Todo: evaluate, select, cross in a loop


        # for i in range(0, self.max_iter):
