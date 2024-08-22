import pandas as pd
from .data import restricted_model_parameters
from .data_generation import generate_data

# Make sure we can see everything
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 300)
pd.set_option('display.width', 1000)

generate_data(restricted_model_parameters, 180)