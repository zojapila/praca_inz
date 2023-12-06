import data_preperation.data_preprocessing as preprocessing
import ga.genetic_algorithm as ga
import pso.pso_algorithm as pso
import testing.pso_testing as test


# training_data = "D:/studia/inzynierka/unsw_nb15/UNSW_NB15_training-set.csv"
work_computer = "C:/praca_inz/source_code/UNSW_NB15_training-set.csv"
testing_data = "C:/praca_inz/UNSW_NB15_testing-set.csv"
pso_results = "C:/praca_inz/source_code/pso_results1.csv"
# processed_data = preprocessing.DataPreprocessing(training_data, alg_type='ga')
processed_data = preprocessing.DataPreprocessing(work_computer)

# ga
# genetic_algorithm = ga.GeneticAlgorithm(processed_data, initial_population_size=200, max_iter=10)
# result = genetic_algorithm.geneticAlgorithmLoop()
# print(result.head())
# pso
# print(processed_data.processed_dataframe.head())
# pso = pso.ParticleSwarmOptimization(processed_data, population_size=50, max_iter=3, final_sol_num=15)
# # pso.normalization()
# pso.algorithmLoop()

results_pso = test.PSOTesting(testing_data, pso_results)
results_pso.testPSO()






