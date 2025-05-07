from implementations.numba.pso.particle import NumbaParticle
import numpy as np
from numba import njit, prange
from core.pso import PSO as BasePSO
from core.functions import objective

@njit(parallel=True, fastmath=True)
def evaluate_pso_numba(positions, function_name):
    n_particles = positions.shape[0]
    costs = np.zeros(n_particles)

    for i in prange(n_particles):
        if function_name in ['ackley', 'rastrigin', 'rosenbrock']:
            costs[i] = objective(positions[i], function_name)
        else:
            raise ValueError(f"Function '{function_name}' not supported.")
    return costs

class NumbaPSO(BasePSO):
    def __init__(self, n_particles, dim, bounds, vel_min, vel_max, w, c1, c2, function:str):
        super().__init__(n_particles, dim, bounds, vel_min, vel_max, w, c1, c2)
        self.function = function

        self.particles = [NumbaParticle(dim, bounds[0], bounds[1], vel_min, vel_max) for _ in range(n_particles)]

        self.initialize(lambda x: objective(x.reshape(1,-1), self.function)[0])

    def move_particles(self, w: float):

        positions = np.array([particle.position for particle in self.particles])

        for particle in self.particles:
            particle.update_velocity(w, self.c1, self.c2, self.gbest)
            particle.update_position(positions)
        
        positions_updated = np.array([particle.position for particle in self.particles])
        costs = evaluate_pso_numba(positions_updated, self.function)

        for i, particle in enumerate(self.particles):
            cost = costs[i]
            if cost < particle.pbest_cost:
                particle.pbest_position = particle.position.copy()
                particle.pbest_cost = cost
            if cost < self.gbest_cost:
                self.gbest = particle.position.copy()
                self.gbest_cost = cost
    
    def optimize(self, max_iterations: int) -> tuple:

        w_min = 0.4
        w_values = np.linspace(self.w_init, w_min, max_iterations)
        
        self.cost_history = []
        for it in range(max_iterations):
            self.move_particles(w_values[it])
            self.cost_history.append(self.gbest_cost)
        
        return self.gbest_cost, self.gbest, self.cost_history
    
    @classmethod
    def is_available(cls) -> bool:
        try:
            import numba
            return True
        except ImportError:
            return False

def my_numba_pso(function: str, bounds: tuple, n_particles: int, iters: int, 
              w: float, c1: float, c2: float, dim: int = 2, 
              vel_min: float = -0.1, vel_max: float = 0.1) -> tuple:
            
    optimizer = NumbaPSO(n_particles, dim, bounds, vel_min, vel_max, w, c1, c2, function)
    cost, pos, cost_history = optimizer.optimize(iters)

    return cost, pos, cost_history