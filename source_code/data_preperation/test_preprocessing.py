from data_preprocessing import *


# training_data = "D:/studia/inzynierka/data/kddcup.data_10_percent.gz"
training_data = "D:/studia/inzynierka/unsw_nb15/UNSW_NB15_training-set.csv"
data = DataPreprocessing()
data.loadDatabase(training_data)
# used_features = ['id', 'srcip', "sport", 'dstip', 'dsport', 'sbytes', 'dbytes', 'state', 'dur', 'proto', 
#                  'service', 'trans_depth', 'attack_cat', 'label']
data.selectFeatures()
# print(data.processed_dataframe.head())
data.convertStringtoInt(True)
data.findMinAndMaxValues()
# print(data.conversion_dicts)
# print(data.min_max_column_vals)
data.removeStringColumns()
print(data.processed_dataframe.head())

