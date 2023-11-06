import random

from data_preperation.data_preprocessing import DataPreprocessing
# import pandas as pd
# import numpy as np


class GeneticAlgorithm:
    def __init__(self, data: DataPreprocessing, initial_population_size: int = 100, max_iter: int = 100) -> None:
        self.population_size: int = initial_population_size
        self.data: DataPreprocessing = data
        self.max_iter = max_iter
        # self.population: list[list] = []
        self.population: dict = {}
        self.evaluation_results: dict = {}
        self.first_free_idx = 0
        # self.generateInitialPopulation()
        # self.evaluationFunction(1)

        # print(self.population)

    def chromosomeCoding(self):
        pass

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
        return True

    def evaluationFunction(self, idx) -> float:
        # w artykule jest dopasowanie tylko jako 0 albo 1 ale chyba trzeba zrobic eksperyment - moze od 1 do 10 wellsee
        # for i in self.data.attack_df
        result = 0
        idxs_to_check = [i for i in range(0, len(self.population[idx]))]
        for i in self.data.attack_df.itertuples():
            for column in idxs_to_check:
                # label jest 4 od konca, przy sprawdzaniu mmusimy to pominać
                # TODO albo column labels niech juz bedzie w ga wypełniane

                if i[column + 2] == self.population[idx][column]:
                    # print(self.data.column_labels[column])
                    result += 1
                    idxs_to_check.remove(column)
            # print(i[])
        # print(result)
        return result

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
        pass

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

    def geneticAlgotrithmLoop(self):
        self.generateInitialPopulation()
        for keys, _ in self.population.items():
            self.evaluation_results[keys] = self.evaluationFunction(keys)

        # for i in range(0, self.max_iter):
