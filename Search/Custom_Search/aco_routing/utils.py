import random
from typing import Dict
# Add caching for edge desirability calculations
_desirability_cache = {}

def compute_edge_desirability(
    pheromone_value: float, 
    edge_cost: float, 
    edge_distance: float,
    alpha: float, 
    beta: float,
    mode: int = 0
) -> float:
    """Compute the desirability with caching for better performance"""
    # Cache key for this computation
    cache_key = (pheromone_value, edge_cost, edge_distance, alpha, beta)
    
    # Return cached result if available
    if cache_key in _desirability_cache:
        return _desirability_cache[cache_key]
        
    # Standard calculation
    if edge_cost == 0:
        edge_cost = 1e-10
    
    # if mode == 2:  # TSP mode
    #     result = (pheromone_value ** alpha) * ((1 / edge_cost ) ** beta)
    # else:
    #     result = (pheromone_value ** alpha) * ((edge_distance / edge_cost) ** beta)
    result = (pheromone_value ** alpha) * ((1 / edge_cost ) ** beta)
    # Store in cache (but limit cache size)
    if len(_desirability_cache) < 10000:  # Prevent memory issues
        _desirability_cache[cache_key] = result
        
    return result

def roulette_wheel_selection(probabilities: Dict[str, float]) -> str:
    """Select a key from a dictionary based on probability values
    
    Args:
        probabilities: Dict mapping items to their probabilities
        
    Returns:
        The selected key
    """
    # Guard against empty dictionary
    if not probabilities:
        return None
        
    # Handle the case where there's only one option
    if len(probabilities) == 1:
        return list(probabilities.keys())[0]
    
    # Ensure probabilities sum to 1
    total = sum(probabilities.values())
    normalized_probs = {k: v/total for k, v in probabilities.items()}
    
    # Pick a random point on the probability line
    r = random.random()
    
    # Find the item that corresponds to this point
    cumulative = 0
    for item, prob in normalized_probs.items():
        cumulative += prob
        if r <= cumulative:
            return item
            
    # Fallback: return random item (should rarely happen)
    return random.choice(list(probabilities.keys()))

def pseudo_random_proportional_selection(value: Dict[str, float], probabilities: Dict[str, float]) -> str:
    """Select a key from a dictionary based on probability values
    
    Args:
        value: Dict mapping items to their value (result of compute_edge_desirability)
        probabilities: Dict mapping items to their probabilities
        
    Returns:
        The selected key or None if no valid options
    """
    # Guard against empty dictionary or non-dictionary input
    if not value or not isinstance(value, dict):
        return None
    
    if len(value) == 0:
        return None
        
    # Handle the case where there's only one option
    if len(value) == 1:
        return list(value.keys())[0]
    
    # Pick a random point on the probability line
    r = random.random() 
    
    threshold = 0.5
    
    # Check if the random number is less than the threshold
    if r < threshold:
        # Select the key with the maximum value
        max_key = max(value, key=value.get)
        return max_key
    else:
        # Perform roulette wheel selection
        selected_key = roulette_wheel_selection(probabilities)
        return selected_key

