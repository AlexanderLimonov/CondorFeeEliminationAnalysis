from itertools import product
from pandas import DataFrame
import pandas as pd
from typing import TypedDict, Any, get_origin, get_args, cast

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
