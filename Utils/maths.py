import math

def flow_to_velocity(flow_rate: float, status: bool = True):
    """
    Convert flow rate to velocity.

    Parameters:
    flow_rate (float): Flow rate in cubic meters per second (vehicle/h).
    distance (float): Distance in meters (m).
    states (bool): True when the flow at each segment is under capacity, False otherwise.

    Returns:
    float: velocity in km/h
    """
    if flow_rate < 0:
        raise ValueError("Flow Rate must be greater than zero.")
    
    # Checking domain of flow rate
    if flow_rate > 1500:
        raise ValueError("Flow Rate must be less than or equal to 1500 vehicle/h.")
    
    # Fixed formula for velocity calculation
    A = -1.4648375
    B = 93.75
    
    if status: # Assuming the flow is under capacity
        vel = (-B - math.sqrt(B**2 + 4 * A * float(flow_rate))) / (2 * A)
    else: # Assuming the flow is over capacity
        vel = (-B + math.sqrt(B**2 + 4 * A * float(flow_rate))) / (2 * A)
        
    # Cap of velocity at 0 km/h and 60 km/h
    vel = max(0, min(60, vel))
    
    return round(vel)

def velocity_to_time(velocity: float, distance: float):
    """
    Convert velocity to time.

    Parameters:
    velocity (float): Velocity in km/h.
    distance (float): Distance in meters (m).

    Returns:
    float: time in seconds
    """
    if velocity <= 0:
        raise ValueError("Velocity must be greater than zero.")
    
    # Convert velocity from km/h to m/s
    velocity_m_s = velocity * 1000 / 3600
    
    # Calculate time in seconds
    time = distance / velocity_m_s
    
    return round(time, 2)  # Round to two decimal places for better readability