import pandas as pd
import pso.pso_algorithm as pso

# df = pd.read_csv('datafile759.csv')
# df_pso =
df = pd.read_csv('D:/studia/inzynierka/source_code/pso_df.csv')
# pso = pso.ParticleSwarmOptimization(processed_data, population_size=20, max_iter=2)
print(df.head())
max_val = df.max().max()
print(max_val)