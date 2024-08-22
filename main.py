from .data import restricted_model_parameters
from .data_generation import parallel_generate_data


if __name__ == "__main__":
    #result = 
    parallel_generate_data(restricted_model_parameters, 10)
    #print(result)