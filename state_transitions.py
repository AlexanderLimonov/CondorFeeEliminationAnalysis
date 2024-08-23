from pandas import DataFrame
from data import ModelInstance, State, model_parameter_names, state_variable_names

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