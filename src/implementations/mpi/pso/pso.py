
from core.functions import objective
import numpy as np
from mpi4py import MPI # type: ignore
from core.pso import PSO as BasePSO
from implementations.mpi.pso.particle import MPIParticle
import pyswarms as ps #type: ignore
from pyswarms.utils.functions import single_obj as fx #type: ignore

class MPIPSO(BasePSO):

    def __init__(self, number_of_particles: int, dim: int, bounds: tuple, vel_min: float, 
                vel_max: float, w: float, c1: float, c2: float,sync_interval:int=5):

        super().__init__(number_of_particles, dim, bounds, vel_min, vel_max, w, c1, c2)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.sync_interval = sync_interval

        particles_per_process = number_of_particles // self.size
        remainder = number_of_particles % self.size

        if self.rank < remainder:
            local_particles = particles_per_process + 1
        else:
            local_particles = particles_per_process
        
        pos_min, pos_max = bounds
        self.particles = [
            MPIParticle(dim, pos_min, pos_max, vel_min, vel_max, self.comm, self.rank) 
            for _ in range(local_particles)
        ]
    
    def initialize(self, objective_function):
        local_best_cost = np.inf
        local_best_position = None

        for particle in self.particles:
            cost = particle.evaluate(objective_function)
            if cost < local_best_cost:
                local_best_cost = cost
                local_best_position = particle.position.copy()

        all_costs = self.comm.allgather(local_best_cost)
        all_positions = self.comm.allgather(local_best_position)

        best_idx = np.argmin(all_costs)
        self.gbest_cost = all_costs[best_idx].copy()
        self.gbest = all_positions[best_idx].copy()

    def move_particles(self, w: float, objective_function):
        
        local_best_cost = self.gbest_cost
        local_best_position = self.gbest.copy() if self.gbest is not None else None

        for particle in self.particles:
            particle.update_velocity(w, self.c1, self.c2, local_best_position)
            particle.update_position()

            cost = particle.evaluate(objective_function)

            if cost < local_best_cost:
                local_best_cost = cost
                local_best_position = particle.position.copy()
        
        return local_best_cost, local_best_position

    def optimize(self, objective_function, num_iterations: int) -> tuple:
        self.initialize(objective_function)
        w_min = 0.4
        w_values = np.linspace(self.w_init, w_min, num_iterations)

        self.cost_history = [self.gbest_cost]

        for it in range(num_iterations):
            local_best_cost,local_best_position = self.move_particles(w_values[it], objective_function)
            if (it + 1) % self.sync_interval == 0 or it == num_iterations - 1:
                all_costs =self.comm.allgather(local_best_cost)
                all_positions = self.comm.allgather(local_best_position)
                best_idx = np.argmin(all_costs)
                global_best_cost = all_costs[best_idx].copy()
                global_best_position = all_positions[best_idx].copy()

                if global_best_cost < self.gbest_cost:
                    self.gbest_cost = global_best_cost.copy()
                    self.gbest = global_best_position.copy()
            self.cost_history.append(self.gbest_cost)

        all_histories = self.comm.gather(self.cost_history)

        combined_history = []
        for i in range(len(self.cost_history)):
            costs_at_iteration = [history[i] for history in all_histories if i < len(history)]
            combined_history.append(min(costs_at_iteration))

        self.cost_history = combined_history

        return self.gbest_cost, self.gbest, self.cost_history

    @classmethod
    def is_available(cls) -> bool:
        try:
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            return size > 1
        except ImportError:
            return False


def my_mpi_pso(function: str, bounds:tuple, n_particles: int, iters: int, w: float, c1: float, c2: float, dim:int=2, vel_min: float=-0.1, vel_max: float=0.1) -> tuple:
    
    # Validar la función objetivo
    if function not in ["ackley", "rastrigin", "rosenbrock"]:
        return None, None, None
       
    # Inicializar el optimizador
    optimizer = MPIPSO(n_particles, dim, bounds, vel_min, vel_max, w, c1, c2)

    # Execute mpi optimization
    cost, pos, cost_history = optimizer.optimize(objective, num_iterations=iters)
    
    return cost, pos, cost_history
    

def pyswarms_pso(function: str, bounds, n_particles, iters, options: dict) -> tuple:
    """PySwarms PSO implementation.
    Args:
        function (str): The objective function to optimize. Supported functions: "ackley", "rastrigin", "rosenbrock".
        bounds (tuple): Tuple containing the minimum and maximum bounds for the position.
        n_particles (int): Number of particles in the swarm.
        iters (int): Number of iterations for the optimization.
        options (dict): Options for PySwarms PSO.
    Returns:
        tuple: Best cost, best position, and cost history.
    """
    # Only supporting 3 functions: ackley, rastrigin and rosenbrock
    if function == "ackley":
        func = fx.ackley
    elif function == "rastrigin":
        func = fx.rastrigin
    elif function == "rosenbrock":
        func = fx.rosenbrock
    else:
        return None, None, None

    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=len(bounds[0]),
        options=options,
        bounds=bounds,
        velocity_clamp=(-0.1, 0.1)
    )
    cost, pos = optimizer.optimize(func, iters=iters)
    cost_history = optimizer.cost_history
    return cost, pos, cost_history


def run_optimization(params_and_function: tuple, rep_num=0):
    """Run the optimization for a given set of parameters and function.
    Args:
        params_and_function (tuple): Tuple containing parameters and function name.
        rep_num (int): Repetition number for the optimization.
    Returns:
        list: List containing the results of the optimization.
    """
    # Disable PySwarms logging completely
    import logging
    logging.getLogger("pyswarms").setLevel(logging.CRITICAL)
    
    # Also disable PySwarms standard output during optimization
    import sys
    import os
    from contextlib import contextmanager
    
    @contextmanager
    def suppress_stdout_stderr():
        # Redirect stdout and stderr to /dev/null
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        
        try:
            yield
        finally:
            # Restore stdout and stderr
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)
    
    # Unpack parameters and function
    n_particles, iters, w, c1, c2, dim, function = params_and_function
    
    # Set bounds based on the function
    if function == "rastrigin":
        bounds = ([-5.12] * dim, [5.12] * dim)
    else:
        bounds = ([-5] * dim, [5] * dim)
    
    # My Async PSO (without internal progress bar)
    time_start1 = time.time()
    my_result = my_mpi_pso(
        function=function, bounds=bounds, n_particles=n_particles, 
        iters=iters, dim=dim, w=w, c1=c1, c2=c2
    )
    my_mpi_pso_time = time.time() - time_start1
    
    # PySwarms
    time_start2 = time.time()
    with suppress_stdout_stderr():
        pyswarms_result = pyswarms_pso(
            function, bounds=bounds, n_particles=n_particles, 
            iters=iters, options={'w': w, 'c1': c1, 'c2': c2}
        )
    pyswarms_pso_time = time.time() - time_start2
    
    if pyswarms_result[0] is None:
        params = {'Number of particles' : n_particles, 'Iterations' : iters, 'w': w, 'c1': c1, 'c2': c2, 'Dimensions': dim, 'Function': function}
        # Error handling for PySwarms
        print(f"Error: PySwarms optimization failed for function {function} with params {params}")
        return None
    ""
    # Create result dictionaries without printing anything
    my_result_dict = {
        'method': 'my_mpi_pso',
        'function': function,
        'n_particles': n_particles,
        'iters': iters,
        'w': w,
        'c1': c1,
        'c2': c2,
        'dim': dim,
        'cost': my_result[0],
        'position': my_result[1].tolist(),
        'execution_time': my_mpi_pso_time,
        'cost_history': my_result[2],
        'repetition': rep_num
    }
    
    pyswarms_result_dict = {
        'method': 'pyswarms_pso',
        'function': function,
        'n_particles': n_particles,
        'iters': iters,
        'w': w,
        'c1': c1,
        'c2': c2,
        'dim': dim,
        'cost': pyswarms_result[0],
        'position': pyswarms_result[1].tolist(),
        'execution_time': pyswarms_pso_time,
        'cost_history': pyswarms_result[2],
        'repetition': rep_num
    }
    
    # Remove all print statements
    
    return [my_result_dict, pyswarms_result_dict]
