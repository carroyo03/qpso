from typing import override
import numpy as np
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
from core.pso import PSO as BasePSO
from implementations.dask.pso.particle import DaskParticle
from core.functions import objective

def evaluate_particle_dask(particle_data, function):
    position = particle_data['position']
    velocity = particle_data['velocity']
    pbest_position = particle_data['pbest_position']
    pbest_cost = particle_data['pbest_cost']

    if function in ['ackley', 'rastrigin', 'rosenbrock']:
        cost = objective(function, position)
    else:
        raise ValueError(f"Unsupported function: {function}")
    
    if cost < pbest_cost:
        particle_data['pbest_position'] = position.copy()
        particle_data['pbest_cost'] = cost
    
    return {
        'position': position,
        'velocity': velocity,
        'pbest_position': pbest_position,
        'pbest_cost': pbest_cost,
        'cost': cost
    }

def update_particles_dask(particle_data, w, c1, c2, gbest_position, pos_min, pos_max, vel_min, vel_max):
    position = particle_data['position']
    velocity = particle_data['velocity']
    pbest_position = particle_data['pbest_position']
    pbest_cost = particle_data['pbest_cost']

    r1 = np.random.random()
    r2 = np.random.random()

    cognitive = c1 * r1 * (pbest_position - position)
    social = c2 * r2 * (gbest_position - position)
    new_velocity = w * velocity + cognitive + social
    velocity = np.clip(new_velocity, vel_min, vel_max)
    position += velocity
    position = np.clip(position, pos_min, pos_max)
    return {
        'position': position,
        'velocity': velocity,
        'pbest_position': pbest_position,
        'pbest_cost': pbest_cost
    }

class DaskPSO(BasePSO):
    def __init__(self, num_particles:int, dim: int,
                 bounds: tuple, vel_min: float, vel_max: float,
                 w: float, c1: float, c2: float, function: str,
                 n_workers: int, scheduler: str = 'threads'):
        super().__init__(num_particles, dim, bounds, vel_min, vel_max, w, c1, c2)
        self.function = function
        self.n_workers = n_workers
        self.scheduler = scheduler
        self.client = None

        if scheduler == 'distributed':
            self.cluster = LocalCluster(n_workers=n_workers)
            self.client = Client(self.cluster)
        self.particles = [
            DaskParticle(dim, bounds[0], bounds[1], vel_min, vel_max)
            for _ in range(num_particles)
        ]

        self.initialize(lambda x: objective(x.reshape(1,-1), self.function)[0])
    
    def __del__(self):
        if self.client is not None:
            self.client.close()
            self.cluster.close()

    def move_particles(self, w: float):
        
        particle_dicts = [
            {
                'position': particle.position,
                'velocity': particle.velocity,
                'pbest_position': particle.pbest_position,
                'pbest_cost': particle.pbest_cost
            }
            for particle in self.particles
        ]

        update_tasks = []
        for particle_dict in particle_dicts:
            updated = dask.delayed(update_particles_dask)(
                particle_dict, w, self.c1, self.c2, self.gbest,
                self.pos_min, self.pos_max, self.vel_min, self.vel_max
            )
            update_tasks.append(updated)
        
        updated_particles = dask.compute(*update_tasks, scheduler=self.scheduler)

        eval_tasks = []
        for particle_dict in updated_particles:
            evaluated = dask.delayed(evaluate_particle_dask)(
                particle_dict, self.function
            )
            eval_tasks.append(evaluated)
        evaluated_particles = dask.compute(*eval_tasks, scheduler=self.scheduler)

        for i, (particle, eval_result) in enumerate(zip(self.particles, evaluated_particles)):
            particle.position = eval_result['position']
            particle.velocity = eval_result['velocity']
            particle.pbest_position = eval_result['pbest_position']
            particle.pbest_cost = eval_result['pbest_cost']
            if eval_result['cost'] < self.gbest_cost:
                self.gbest = particle.position.copy()
                self.gbest_cost = eval_result['cost']
    
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
            import dask
            return True
        except ImportError:
            return False

def my_dask_pso(function: str, bounds: tuple, n_particles: int, iters: int, 
             w: float, c1: float, c2: float, dim: int = 2, 
             vel_min: float = -0.1, vel_max: float = 0.1,
             n_workers: int = None, scheduler: str = 'threads') -> tuple:

    optimizer = DaskPSO(n_particles, dim, bounds, vel_min, vel_max, w, c1, c2, function, n_workers, scheduler)

    cost, pos, cost_history = optimizer.optimize(iters)
    return cost, pos, cost_history

