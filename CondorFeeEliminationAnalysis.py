import pandas as pd
from pandas import DataFrame
from typing import TypedDict, Any, get_origin, get_args, cast
from itertools import product

# Constants and variable ranges

MOTES_GAS_MIN = list(range(1, 6))
MOTES_GAS_MAX = [5]
MOTES_CSPR = [1_000_000_000]
GAS_PER_BLOCK = [4_000_000_000_000]
SECONDS_PER_BLOCK = [8, 16]
DAYS_TO_RELEASE = list(range(0, 181))
AMORTIZED = [False, True]
GAS_COST_CONTRACT_CALL = [100_000_000_000]

CSPR_PRICE = [0.02]
DAILY_INTEREST = [0.1/365]

BLOCK_UTILIZATION = [0.2]
UNSTAKED_CSPR = [4_000_000_000]
PRICE_ELASTICITY = [1.5]
UTILIZATION_LOWER_THRESHOLD = [0.5]
UTILIZATION_HIGHER_THRESHOLD = [0.9]

ATTACK_UTILIZATION = [1.0]
ATTACKER_BUDGET = [500_000.0, 1_000_000.0, 2_000_000.0, 3_000_000.0, 4_000_000.0, 5_000_000_000.0]

INITIAL_CAPITAL = [50_000.0]
CUSTOMERS = [50]
INFRA_COST = [10.0]

SUBSCRIPTION = [1.3]
CONTRACT_CALLS = [15]
CONTRACT_CALLS_TO_DROP = [2]

FIRM_SCALE = [1, 2, 3, 4, 5]

class ModelParameters(TypedDict):
    motes_gas_min: list[int]
    motes_gas_max: list[int]
    motes_CSPR: list[int]
    gas_per_block: list[int]
    seconds_per_block: list[int]
    days_to_release: list[int]
    amortized: list[bool]
    gas_per_contract_call: list[int]
    CSPR_price: list[float]
    daily_interest: list[float]
    block_utilization: list[float]
    unstaked_motes: list[int]
    price_elasticity: list[float]
    utilization_lower_threshold: list[float]
    utilization_higher_threshold: list[float]
    attack_utilization: list[float]
    attacker_budget_USD: list[float]
    initial_capital_USD: list[float]
    customers: list[int]
    infra_cost_USD: list[float]
    subscription_USD: list[float]
    contract_calls: list[int]
    contract_calls_to_drop: list[int]
    firm_scale: list[int]
        
def model_parameter_names() -> list[str]:
    return list(ModelParameters.__annotations__.keys())

default_model_parameters: ModelParameters = {
        'motes_gas_min': MOTES_GAS_MIN,
        'motes_gas_max': MOTES_GAS_MAX,
        'motes_CSPR': MOTES_CSPR,
        'gas_per_block': GAS_PER_BLOCK,
        'seconds_per_block': SECONDS_PER_BLOCK,
        'days_to_release': DAYS_TO_RELEASE,
        'amortized': AMORTIZED,
        'gas_per_contract_call': GAS_COST_CONTRACT_CALL,
        'CSPR_price': CSPR_PRICE,
        'daily_interest': DAILY_INTEREST,
        'block_utilization': BLOCK_UTILIZATION,
        'unstaked_motes': UNSTAKED_CSPR,
        'price_elasticity': PRICE_ELASTICITY,
        'utilization_lower_threshold': UTILIZATION_LOWER_THRESHOLD,
        'utilization_higher_threshold': UTILIZATION_HIGHER_THRESHOLD,
        'attack_utilization': ATTACK_UTILIZATION,
        'attacker_budget_USD': ATTACKER_BUDGET,
        'initial_capital_USD': INITIAL_CAPITAL,
        'customers': CUSTOMERS,
        'infra_cost_USD': INFRA_COST,
        'subscription_USD': SUBSCRIPTION,
        'contract_calls': CONTRACT_CALLS,
        'contract_calls_to_drop': CONTRACT_CALLS_TO_DROP,
        'firm_scale': FIRM_SCALE,
    }

restricted_model_parameters: ModelParameters = {
        'motes_gas_min': MOTES_GAS_MIN,
        'motes_gas_max': MOTES_GAS_MAX,
        'motes_CSPR': MOTES_CSPR,
        'gas_per_block': GAS_PER_BLOCK,
        'seconds_per_block': SECONDS_PER_BLOCK,
        'days_to_release': list(range(0, 14)),
        'amortized': AMORTIZED,
        'gas_per_contract_call': GAS_COST_CONTRACT_CALL,
        'CSPR_price': CSPR_PRICE,
        'daily_interest': DAILY_INTEREST,
        'block_utilization': BLOCK_UTILIZATION,
        'unstaked_motes': UNSTAKED_CSPR,
        'price_elasticity': PRICE_ELASTICITY,
        'utilization_lower_threshold': UTILIZATION_LOWER_THRESHOLD,
        'utilization_higher_threshold': UTILIZATION_HIGHER_THRESHOLD,
        'attack_utilization': ATTACK_UTILIZATION,
        'attacker_budget_USD': [1_000_000.0],
        'initial_capital_USD': INITIAL_CAPITAL,
        'customers': CUSTOMERS,
        'infra_cost_USD': INFRA_COST,
        'subscription_USD': SUBSCRIPTION,
        'contract_calls': CONTRACT_CALLS,
        'contract_calls_to_drop': CONTRACT_CALLS_TO_DROP,
        'firm_scale': FIRM_SCALE,
    }

class ModelInstance(TypedDict):
    motes_gas_min: int
    motes_gas_max: int
    motes_CSPR: int
    gas_per_block: int
    seconds_per_block: int
    days_to_release: int
    amortized: bool
    gas_per_contract_call: int
    CSPR_price: float
    daily_interest: float
    block_utilization: float
    unstaked_motes: int
    price_elasticity: float
    utilization_lower_threshold: float
    utilization_higher_threshold: float
    attack_utilization: float
    attacker_budget_USD: int
    initial_capital_USD: int
    customers: int
    infra_cost_USD: float
    subscription_USD: float
    contract_calls: int
    contract_calls_to_drop: int

class State(TypedDict):
    day: int
    gas_price: int
    available_gas: int
    attacker_budget_motes: int
    attacker_target_utilization: float
    attacker_realized_utilization: float
    attacker_held_motes: int
    attacker_released_motes: int
    users_target_utilization: float
    users_realized_utilization:float
    users_held_motes: int
    users_released_motes: int
    firm_customers: int
    firm_budget_motes: int
    firm_target_calls: int
    firm_realized_calls: int
    firm_revenue_motes: int
    firm_cost_motes: int
    firm_rewards_motes: int
    firm_held_motes: int
    firm_released_motes: int
    unused_gas: int
    
def state_variable_names() -> list[str]:
    return list(State.__annotations__.keys())
    
def init_state(instance: ModelInstance) -> State:
    initial_available_gas = int(((24*60*60)/instance['seconds_per_block'])*instance['gas_per_block'])
    initial_state: State = {
        'day': 1, 
        'gas_price': instance['motes_gas_min'],
        'available_gas': initial_available_gas,
        'attacker_budget_motes': int((instance['attacker_budget_USD']/instance['CSPR_price'])*instance['motes_CSPR']),
        'attacker_target_utilization': 0.0,
        'attacker_realized_utilization': 0.0,
        'attacker_held_motes': 0,
        'attacker_released_motes': 0,
        'users_target_utilization': 0.0,
        'users_realized_utilization': 0.0,
        'users_held_motes': 0,
        'users_released_motes': 0,
        'firm_customers': instance['customers'],
        'firm_budget_motes': int((instance['initial_capital_USD']/instance['CSPR_price'])*instance['motes_CSPR']),
        'firm_target_calls': 0,
        'firm_realized_calls': 0,
        'firm_revenue_motes': 0,
        'firm_cost_motes': 0,
        'firm_rewards_motes': 0,
        'firm_held_motes': 0,
        'firm_released_motes': 0,
        'unused_gas': initial_available_gas,
    }
    
    return initial_state

# Helpers to extract field and type information
# Function to map Python types to pandas dtypes
def get_pandas_dtype(py_type: Any) -> str:
    if py_type == int:
        return 'int64'
    elif py_type == float:
        return 'float64'
    elif py_type == bool:
        return 'bool'
    elif py_type == str:
        return 'object'
    elif get_origin(py_type) == list:
        # Extract the type of elements in the list
        elem_type = get_args(py_type)[0]
        return get_pandas_dtype(elem_type)
    else:
        return 'object'

# Abstract definitions of the recursive DataFrame row generation
def system_state_transition(max_day: int, state_frame: DataFrame) -> DataFrame:
    last_row_index: int = state_frame.index.max() # type: ignore

    instance: ModelInstance = state_frame[-1:][model_parameter_names()].to_dict(orient = 'index')[last_row_index] # type: ignore
    last_state: State = state_frame[-1:][state_variable_names()].to_dict(orient = 'index')[last_row_index] # type: ignore
    
    attacker_released_motes = 0
    firm_released_motes = 0

    # Check for token releases
    if instance['days_to_release'] == 0:
        attacker_released_motes: int = state_frame.loc[last_row_index, 'attacker_held_motes'] # type: ignore
        firm_released_motes: int = state_frame.loc[last_row_index, 'firm_held_motes'] # type: ignore
        
        state_frame.loc[last_row_index, 'attacker_released_motes'] = attacker_released_motes
        state_frame.loc[last_row_index, 'users_released_motes'] = state_frame.loc[last_row_index, 'users_held_motes'] # type: ignore
        state_frame.loc[last_row_index, 'firm_released_motes'] = firm_released_motes
    elif instance['amortized'] == True:
        days_to_release = instance['days_to_release']
        release_multiplier = 1/days_to_release
            
        index_start = last_row_index - days_to_release # type: ignore
        index_end = last_row_index - 1 # type: ignore
            
        attacker_released_motes: int = (state_frame.loc[index_start:index_end, 'attacker_held_motes']*release_multiplier).astype(dtype = 'int64').sum() # type: ignore
        firm_released_motes: int = (state_frame.loc[index_start:index_end, 'firm_held_motes']*release_multiplier).astype(dtype = 'int64').sum() # type: ignore

        state_frame.loc[last_row_index, 'attacker_released_motes'] = attacker_released_motes
        state_frame.loc[last_row_index, 'users_released_motes'] = (state_frame.loc[index_start:index_end, 'users_held_motes']*release_multiplier).astype(dtype = 'int64').sum() # type: ignore
        state_frame.loc[last_row_index, 'firm_released_motes'] = firm_released_motes
    else:
        days_to_release = instance['days_to_release']
            
        index = last_row_index - days_to_release # type: ignore
           
        if index >= 0:
            attacker_released_motes: int = state_frame.loc[index, 'attacker_held_motes'] # type: ignore
            firm_released_motes: int = state_frame.loc[index, 'firm_held_motes'] # type: ignore

            state_frame.loc[last_row_index, 'attacker_released_motes'] = attacker_released_motes
            state_frame.loc[last_row_index, 'users_released_motes'] = state_frame.loc[index, 'users_held_motes'] # type: ignore
            state_frame.loc[last_row_index, 'firm_released_motes'] = firm_released_motes
    
    last_day = last_state['day']
    
    firm_remaining_budget = max(0, last_state['firm_budget_motes'] - last_state['firm_cost_motes'] + last_state['firm_revenue_motes'])    
    firm_rewards = int(firm_remaining_budget*instance['daily_interest'])
    state_frame.loc[last_row_index, 'firm_rewards_motes'] = firm_rewards

    if last_day < max_day:
        utilization_lower_threshold = instance['utilization_lower_threshold']
        utilization_higher_threshold = instance['utilization_higher_threshold']
        
        initial_available_gas = int(((24*60*60)/instance['seconds_per_block'])*instance['gas_per_block'])
        motes_gas_min = instance['motes_gas_min']
        motes_gas_max = instance['motes_gas_max']
        last_gas_price = last_state['gas_price']
        
        utilization = 1.0 - last_state['unused_gas']/initial_available_gas
        if (utilization >= utilization_lower_threshold) & (utilization <= utilization_higher_threshold):
            new_gas_price = last_gas_price
        elif (utilization < utilization_lower_threshold) & (last_gas_price > motes_gas_min):
            new_gas_price = last_gas_price - 1
        elif (utilization > utilization_higher_threshold) & (last_gas_price < motes_gas_max):
            new_gas_price = last_gas_price + 1
        else:
            new_gas_price = last_gas_price
        
        # Consider setting the budgets in respective functions
        new_state: State = {
            'day': last_day + 1, 
            'gas_price': new_gas_price,
            'available_gas': initial_available_gas,
            'attacker_budget_motes': last_state['attacker_budget_motes'] + attacker_released_motes - last_state['attacker_held_motes'],
            'attacker_target_utilization': 0.0,
            'attacker_realized_utilization': 0.0,
            'attacker_held_motes': 0,
            'attacker_released_motes': 0,
            'users_target_utilization': 0.0,
            'users_realized_utilization': 0.0,
            'users_held_motes': 0,
            'users_released_motes': 0,
            'firm_customers': 0,
            'firm_budget_motes': last_state['firm_budget_motes'] + firm_released_motes - last_state['firm_held_motes'] + firm_rewards,
            'firm_target_calls': 0,
            'firm_realized_calls': 0,
            'firm_revenue_motes': 0,
            'firm_cost_motes': 0,
            'firm_rewards_motes': 0,
            'firm_held_motes': 0,
            'firm_released_motes': 0,
            'unused_gas': initial_available_gas
        }
        
        state_frame.loc[last_row_index + 1] = instance | new_state # type: ignore
                
    return state_frame # type: ignore

def attacker_state_transition(state_frame: DataFrame) -> DataFrame:
    last_row_index: int = state_frame.index.max() # type: ignore
    last_state: State = state_frame[-1:][state_variable_names()].to_dict(orient = 'index')[last_row_index] # type: ignore
    
    gas_price = last_state['gas_price']
    
    available_gas = last_state['unused_gas']
    # if last_row_index == 0:
    attacker_budget = last_state['attacker_budget_motes']
    # else:
    #     before_last_state: State = state_frame[-2:-1][state_variable_names()].to_dict(orient = 'index')[last_row_index - 1] # type: ignore
    #     attacker_budget = before_last_state['attacker_budget_motes'] - before_last_state['attacker_locked_motes'] + before_last_state['attacker_released_motes']
        
    #     state_frame.loc[last_row_index, 'attacker_budget_motes'] = attacker_budget

    affordable_gas = attacker_budget/gas_price
    consumed_gas = min(affordable_gas, available_gas)
    held_motes = int(consumed_gas*gas_price)
    target_utilization = min(1.0, affordable_gas/available_gas)
    realized_utilization = consumed_gas/available_gas
    unused_gas = int(available_gas - consumed_gas)
        
    state_frame.loc[last_row_index, 'attacker_target_utilization'] = target_utilization
    state_frame.loc[last_row_index, 'attacker_realized_utilization'] = realized_utilization
    state_frame.loc[last_row_index, 'attacker_held_motes'] = held_motes
    state_frame.loc[last_row_index, 'unused_gas'] = unused_gas

    return state_frame

def users_state_transition(state_frame: DataFrame) -> DataFrame:
    last_row_index: int = state_frame.index.max() # type: ignore
    instance: ModelInstance = state_frame[-1:][model_parameter_names()].to_dict(orient = 'index')[last_row_index] # type: ignore
    last_state: State = state_frame[-1:][state_variable_names()].to_dict(orient = 'index')[last_row_index] # type: ignore
    
    price_elasticity = instance['price_elasticity']
    demand_multiplier = instance['block_utilization']**(1/price_elasticity)
    
    gas_price = last_state['gas_price']
    available_gas = last_state['available_gas']
    unused_gas = last_state['unused_gas']

    target_utilization = (demand_multiplier/gas_price)**price_elasticity
    realized_utilization = min(unused_gas/available_gas, target_utilization)
    used_gas = realized_utilization*available_gas
    held_motes = used_gas*gas_price
    new_unused_gas = unused_gas - used_gas
    
    state_frame.loc[last_row_index, 'users_target_utilization'] = target_utilization
    state_frame.loc[last_row_index, 'users_realized_utilization'] = realized_utilization
    state_frame.loc[last_row_index, 'users_held_motes'] = held_motes
    state_frame.loc[last_row_index, 'unused_gas'] = new_unused_gas

    return state_frame

def firm_state_transition(state_frame: DataFrame) -> DataFrame:
    last_row_index: int = state_frame.index.max() # type: ignore
    instance: ModelInstance = state_frame[-1:][model_parameter_names()].to_dict(orient = 'index')[last_row_index] # type: ignore
    last_state: State = state_frame[-1:][state_variable_names()].to_dict(orient = 'index')[last_row_index] # type: ignore
    
    gas_price = last_state['gas_price']
    available_gas = last_state['unused_gas']
    
    firm_budget = last_state['firm_budget_motes']
    
    # Customer dropout
    if last_row_index > 0:
        before_last_state: State = state_frame[-2:-1][state_variable_names()].to_dict(orient = 'index')[last_row_index - 1] # type: ignore
        previous_customers = before_last_state['firm_customers']
        missed_calls = before_last_state['firm_target_calls'] - before_last_state['firm_target_calls']
        if missed_calls > instance['contract_calls_to_drop']:
            current_customers = previous_customers - 1
        else:
            current_customers = previous_customers
    else:
        current_customers = last_state['firm_customers']
    
    # Contract calls
    gas_per_call = instance['gas_per_contract_call']
    affordable_gas = firm_budget/gas_price
    target_calls = current_customers*instance['contract_calls']
    consumable_gas = min(affordable_gas, available_gas)
    realized_calls = min(target_calls, int(consumable_gas/gas_per_call))
    consumed_gas = realized_calls*gas_per_call
    held_motes = int(consumed_gas*gas_price)
    new_unused_gas = int(available_gas - consumed_gas)
    
    # Revenue & expenses
    infra_cost_motes = int((instance['infra_cost_USD']/instance['CSPR_price'])*instance['motes_CSPR'])
    subscription_revenue_motes = int((instance['subscription_USD']*current_customers/instance['CSPR_price'])*instance['motes_CSPR'])
    #rewards = int(firm_budget*instance['daily_interest'])

    state_frame.loc[last_row_index, 'firm_customers'] = current_customers
    state_frame.loc[last_row_index, 'firm_target_calls'] = target_calls
    state_frame.loc[last_row_index, 'firm_realized_calls'] = realized_calls
    state_frame.loc[last_row_index, 'firm_revenue_motes'] = subscription_revenue_motes
    state_frame.loc[last_row_index, 'firm_cost_motes'] = infra_cost_motes
    state_frame.loc[last_row_index, 'firm_held_motes'] = held_motes
    state_frame.loc[last_row_index, 'unused_gas'] = new_unused_gas

    return state_frame

def state_transition(max_day: int, state_frame: DataFrame) -> DataFrame:
    updated_state_frame = system_state_transition(max_day, 
                            firm_state_transition(
                                users_state_transition(
                                    attacker_state_transition(state_frame)
                                )
                            )
                          )

    return updated_state_frame      

def init_state_dataframe() -> DataFrame:
     model_instance_fields = ModelParameters.__annotations__ 
     model_instance_columns = {field: get_pandas_dtype(model_instance_fields[field]) for field in model_instance_fields}
     
     state_fields = State.__annotations__
     state_columns = {field: get_pandas_dtype(state_fields[field]) for field in state_fields}
     
     columns = model_instance_columns | state_columns
     
     return DataFrame({col: pd.Series(dtype=typ) for col, typ in columns.items()})

def parameters_instance_valid(instance: ModelInstance) -> bool:
    return (
        instance['motes_gas_min'] <= instance['motes_gas_max'] and
        instance['utilization_lower_threshold'] <= instance['utilization_higher_threshold']
    )

def generate_model_instances(parameters: ModelParameters) -> list[ModelInstance]:
    names: list[str] = list(parameters.keys())
    values: list[Any] = list(parameters.values())

    combinations: list[tuple[Any, ...]] = list(product(*values))
    instances: list[ModelInstance] = []
    for combo in combinations:
        combination_dict: ModelInstance = cast(ModelInstance, dict(zip(names, combo)))
        if parameters_instance_valid(combination_dict):
            instances.append(combination_dict)
        
    return instances


import time
def generate_data(parameters: ModelParameters, days: int) -> DataFrame:
        instances = generate_model_instances(parameters)
        
        state_frames: list[DataFrame] = []
        
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
        print(end)
            
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
from concurrent.futures import ProcessPoolExecutor, as_completed
def parallel_generate_data(parameters: ModelParameters, days: int) -> DataFrame:
   instances = generate_model_instances(parameters)
   
   start = time.perf_counter()
   with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_instance, instance, days) for instance in instances]
        
    state_frames = []
    for future in as_completed(futures):
        state_frames.append(future.result()) # type: ignore
    
    end = time.perf_counter() - start
    print(end)
    print(len(state_frames))

    models_frame = pd.concat(state_frames) # type: ignore
    print(models_frame.shape)
    return models_frame

small_model_parameters: ModelParameters = {
        'motes_gas_min': [1],
        'motes_gas_max': [3],
        'motes_CSPR': MOTES_CSPR,
        'gas_per_block': GAS_PER_BLOCK,
        'seconds_per_block': [16],
        'days_to_release': [5],
        'amortized': AMORTIZED,
        'gas_per_contract_call': GAS_COST_CONTRACT_CALL,
        'CSPR_price': CSPR_PRICE,
        'daily_interest': DAILY_INTEREST,
        'block_utilization': BLOCK_UTILIZATION,
        'unstaked_motes': UNSTAKED_CSPR,
        'price_elasticity': PRICE_ELASTICITY,
        'utilization_lower_threshold': UTILIZATION_LOWER_THRESHOLD,
        'utilization_higher_threshold': UTILIZATION_HIGHER_THRESHOLD,
        'attack_utilization': ATTACK_UTILIZATION,
        'attacker_budget_USD': [1_000_000.0],
        'initial_capital_USD': INITIAL_CAPITAL,
        'customers': CUSTOMERS,
        'infra_cost_USD': INFRA_COST,
        'subscription_USD': SUBSCRIPTION,
        'contract_calls': CONTRACT_CALLS,
        'contract_calls_to_drop': CONTRACT_CALLS_TO_DROP,
        'firm_scale': [1],
    }

# Make sure we can see everything
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 300)
pd.set_option('display.width', 1000)

if __name__ == "__main__":
    #result = 
    parallel_generate_data(restricted_model_parameters, 10)
    #print(result)