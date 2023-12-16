# import data_preperation.data_preprocessing as preprocessing
# import ga.genetic_algorithm as ga
# import pso.pso_algorithm as pso
# import testing.pso_testing as test
# import time
#
#
# training_data = "D:/studia/inzynierka/unsw_nb15/UNSW_NB15_training-set.csv"
# work_computer = "C:/praca_inz/source_code/UNSW_NB15_training-set.csv"
# testing_data = "C:/praca_inz/UNSW_NB15_testing-set.csv"
# pso_results = "C:/praca_inz/source_code/pso_results35.csv"
# # processed_data = preprocessing.DataPreprocessing()
# processed_data = preprocessing.DataPreprocessing(work_computer)
#
# # ga
# # genetic_algorithm = ga.GeneticAlgorithm(processed_data, initial_population_size=300, max_iter=10, num_of_final_sol=50)
# # result = genetic_algorithm.geneticAlgorithmLoop()
# # print(result.head())
# # pso
# # print(processed_data.processed_dataframe.head())
# start = time.time()
# pso = pso.ParticleSwarmOptimization(processed_data, population_size=100, max_iter=15, final_sol_num=100, filenum=35)
# print(pso.attack_df.head())
# # # pso.normalization()
# pso.algorithmLoop()
# end = time.time()
# print(end - start)
# results_pso = test.PSOTesting(testing_data, pso_results)
# results_pso.testPSO()
import data_preperation.data_preprocessing as preprocessing
import ga.genetic_algorithm as ga
import testing.ga_testing as gatest
import pso.pso_algorithm as pso
import testing.pso_testing as test
import time


# training_data = "D:/studia/inzynierka/unsw_nb15/UNSW_NB15_training-set.csv"
work_computer = "C:/praca_inz/source_code/UNSW_NB15_training-set.csv"
testing_data = "C:/praca_inz/UNSW_NB15_testing-set.csv"
pso_results = "C:/praca_inz/source_code/ga_results10.csv"
processed_data = preprocessing.DataPreprocessing(work_computer, alg_type='ga')
# processed_data = preprocessing.DataPreprocessing(work_computer)

# ga
start = time.time()
genetic_algorithm = ga.GeneticAlgorithm(processed_data, initial_population_size=300, max_iter=5,
                                        num_of_final_sol=300, filenum=10)
result = genetic_algorithm.geneticAlgorithmLoop()
end = time.time()
print(end - start)
print(result.head())
res_ga = gatest.GATesting(testing_data, pso_results)
res_ga.testGA()
# pso
# print(processed_data.processed_dataframe.head())
# start = time.time()
# pso = pso.ParticleSwarmOptimization(processed_data, population_size=90, max_iter=10, final_sol_num=50, filenum=28)
# print(pso.attack_df.head())
# # # pso.normalization()
# pso.algorithmLoop()
# end = time.time()
# print(end - start)
# results_pso = test.PSOTesting(testing_data, pso_results)
# results_pso.testPSO()
