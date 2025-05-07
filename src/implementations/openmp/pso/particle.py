import numpy as np
from core.particle import Particle as BaseParticle

try:
    import cpp.pso_core as pso_core  # type: ignore
    HAS_CPP_EXTENSION = True
except ImportError:
    HAS_CPP_EXTENSION = False
    print("C++ extension not found. Using Python implementation instead.")

class Particle(BaseParticle):

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