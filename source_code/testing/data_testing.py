import data_preperation.data_preprocessing as data_prep


class PSOTesting:
    def __init__(self, test_df_path: str, pso_results):
        self.test_df = data_prep.DataPreprocessing(test_df_path)
        self.pso_results = pso_results


'''
plan jest taki, że pso_results to df z wszystkimi parametrami z oryginalnej df oraz r. 
jak już to mam to dla każdego elementu z bazy danych obliczam odległości do każdego wyniku i jesli jest < r to anomalia
później należałoby wyznaczyć sobie skuteczność tego
'''

