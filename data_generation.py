import pandas as pd
from pandas import DataFrame
from concurrent.futures import ProcessPoolExecutor, as_completed
from .data import ModelParameters, ModelInstance, generate_model_instances, init_state_dataframe, init_state
from .state_transitions import state_transition


import time
def generate_data(parameters: ModelParameters, days: int) -> DataFrame:
        instances = generate_model_instances(parameters)
        state_frames: list[DataFrame] = []
        
        print("Started data generation")
        start = time.perf_counter()

        for instance in instances:
            state_frame = init_state_dataframe()
            initial_state = init_state(instance)
            
            # Note the annotation suppressing a type check, necessary here because it's impossible to make pandas work
            # with strict type checking in a straightforward way here
            state_frame.loc[0] = instance | initial_state # type: ignore
            
            updated_state_frame = state_frame # type: ignore
            for _ in range(1, days + 1):
                updated_state_frame = state_transition(days, updated_state_frame) # type: ignore
            
            state_frames.append(updated_state_frame) # type: ignore
            
        end = time.perf_counter() - start
        print("Time to generate " + str(len(state_frames)) + " frames: " + str(end))
            
        models_frame = pd.concat(state_frames) # type: ignore

        return models_frame # type: ignore

# Helper function to be executed in parallel
def process_instance(instance: ModelInstance, days: int) -> DataFrame:
    state_frame = init_state_dataframe()
    initial_state = init_state(instance)
    
    state_frame.loc[0] = instance | initial_state # type: ignore
    
    updated_state_frame = state_frame # type: ignore
    for _ in range(1, days + 1):
        updated_state_frame = state_transition(days, updated_state_frame) # type: ignore
    
    return updated_state_frame

# Parallel version of generate_data
def parallel_generate_data(parameters: ModelParameters, days: int) -> DataFrame:
   instances = generate_model_instances(parameters)
   state_frames: list[DataFrame] = []

   print("Started parallelized data generation")
   start = time.perf_counter()
   
   with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_instance, instance, days) for instance in instances]
        
    for future in as_completed(futures):
        state_frames.append(future.result()) # type: ignore
    
    end = time.perf_counter() - start
    print("Time to generate " + str(len(state_frames)) + " frames in parallel: " + str(end))

    models_frame = pd.concat(state_frames) # type: ignore
    print(models_frame.shape)
    return models_frame

