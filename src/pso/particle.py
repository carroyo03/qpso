import numpy as np

try:
    import cpp.pso_core as pso_core  # type: ignore
    HAS_CPP_EXTENSION = True
except ImportError:
    HAS_CPP_EXTENSION = False
    print("C++ extension not found. Using Python implementation instead.")

class Particle:
    """A simple class to represent a particle in the Particle Swarm Optimization (PSO) algorithm.
    
    Each particle has a position, velocity, and best position found so far.
    
    """
    
    def __init__(self, dim: int, pos_min: np.ndarray, pos_max: np.ndarray, 
                 vel_min: float, vel_max: float):
        """Initialize a particle with random position and velocity.
        
        Args:
            dim (int): Number of dimensions for the particle.
            pos_min (np.ndarray): Minimum bounds for position.
            pos_max (np.ndarray): Maximum bounds for position.
            vel_min (float): Minimum velocity.
            vel_max (float): Maximum velocity.
        """
        # Initialize position randomly within bounds
        self.position = np.random.uniform(low=pos_min, high=pos_max, size=dim)
        
        # Initialize velocity randomly (typically smaller than position range)
        self.velocity = np.random.uniform(
            low=-0.1 * (pos_max - pos_min),
            high=0.1 * (pos_max - pos_min),
            size=dim
        )
        
        # Initialize personal best
        self.pbest_position = self.position.copy()
        self.pbest_cost = float('inf')  # Will be updated on first evaluation
        
        # Store bounds for clipping
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.vel_min = vel_min
        self.vel_max = vel_max
    
    def update_velocity(self, w: float, c1: float, c2: float, gbest_position: np.ndarray):
        """Update the particle's velocity using the PSO velocity equation.
        
        Args:
            w (float): Inertia weight.
            c1 (float): Cognitive coefficient.
            c2 (float): Social coefficient.
            gbest_position (np.ndarray): Global best position.
        """
        # Random components for stochastic behavior
        r1 = np.random.random()
        r2 = np.random.random()
        
        # Cognitive component (attraction to personal best)
        cognitive = c1 * r1 * (self.pbest_position - self.position)
        
        # Social component (attraction to global best)
        social = c2 * r2 * (gbest_position - self.position)
        
        # Update velocity with inertia, cognitive, and social components
        self.velocity = w * self.velocity + cognitive + social
        
        # Clip velocity to bounds
        self.velocity = np.clip(self.velocity, self.vel_min, self.vel_max)
    
    def update_position(self):
        """Update the particle's position based on its velocity.
        """
        # Update position
        self.position += self.velocity
        
        # Clip position to bounds
        self.position = np.clip(self.position, self.pos_min, self.pos_max)
    
    def evaluate(self, objective_function, function_name: str = ""):
        """Evaluate the particle's position and update personal best if improved.
        
        Args:
            objective_function: Function to evaluate the particle's position.
            
        Returns:
            float: The cost at the current position.
        """
        if HAS_CPP_EXTENSION and function_name != "":
            
            cost_array = pso_core.evaluate_swarm(self.position, function_name)
            cost = float(cost_array[0])
        
        else:
            # If not C++, Python implementation is used
        # Evaluate current position
            cost = objective_function(self.position)
        
        # Update personal best if current position is better
        if cost < self.pbest_cost:
            self.pbest_position = self.position.copy()
            self.pbest_cost = cost
            
        return cost