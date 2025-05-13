import numpy as np
import asyncio
from core.pso import PSO as BasePSO
from implementations.async_prog.pso.particle import Particle as AsyncParticle
from core.functions import objective

class AsyncPSO(BasePSO):
    def __init__(self, number_of_particles:int, dim:int, bounds:tuple, vel_min:float, vel_max:float, w:float, c1:float,c2:float):
        """
        Initializes the asynchronous Particle Swarm Optimization (PSO) algorithm instance by
        defining parameters for particles, dimensions, velocity, and various coefficients
        used in the PSO algorithm. It also creates a list of particles for optimization.

        Attributes:
        number_of_particles: int
            The total number of particles in the swarm.
        dim: int
            The number of dimensions for the search space.
        bounds: tuple
            A tuple representing the boundaries of the search space as (min, max).
        vel_min: float
            The minimum velocity allowed for a particle.
        vel_max: float
            The maximum velocity allowed for a particle.
        w: float
            The inertia weight used to control the impact of the previous velocity during
            velocity update.
        c1: float
            The cognitive coefficient, which determines the influence of a particle's personal
            best position on its velocity.
        c2: float
            The social coefficient, which determines the influence of the global best position
            on a particle's velocity.
        particles: list
            A list of instances of AsyncParticle, representing the particles in the swarm.

        Args:
        number_of_particles: int
            The total number of particles to initialize in the swarm.
        dim: int
            The number of dimensions for the optimization problem or search space.
        bounds: tuple
            Range of boundaries specifying the limit for particle positions.
        vel_min: float
            Minimum velocity a particle can have in any dimension.
        vel_max: float
            Maximum velocity a particle can have in any dimension.
        w: float
            Inertia weight used in the velocity update equation.
        c1: float
            The coefficient factor for the cognitive component in the velocity update.
        c2: float
            The coefficient factor for the social component in the velocity update.
        """
        super().__init__(number_of_particles, dim, bounds, vel_min, vel_max, w, c1, c2)
        pos_min, pos_max = bounds
        self.particles = [
            AsyncParticle(dim = dim, pos_min=pos_min, pos_max=pos_max, vel_min = vel_min,
                          vel_max=vel_max) for _ in range(number_of_particles)
        ]

    async def initialize_async(self, objective_function):
        """
        Asynchronously initializes the PSO by evaluating each particle's
        objective function and determining the global best particle's position and cost.

        Args:
            objective_function: A callable function that evaluates the fitness or cost of a
                particle's position in the search space.


        """
        tasks = [particle.evaluate_async(objective_function) for particle in self.particles]
        costs = await asyncio.gather(*tasks)
        best_idx = np.argmin(costs)
        self.gbest = self.particles[best_idx].pbest_position.copy()
        self.gbest_cost = costs[best_idx]
    async def move_particles_async(self, w, objective_function):
        """
        Asynchronously moves particles in the swarm, updates their velocity and position,
        evaluates their cost using the provided objective function, and determines the global
        best position and cost in the swarm.

        Args:
            w: Inertia weight factor that controls the influence of previous velocity on the
               particle's updated velocity.
            objective_function: A callable representing the objective function to evaluate
                                the particles' positions asynchronously.


        """
        _ = [particle.update_velocity_and_position(w, self.c1, self.c2, self.gbest) for particle in self.particles]
        tasks = [particle.evaluate_async(objective_function) for particle in self.particles]
        costs = await asyncio.gather(*tasks)
        best_idx = np.argmin(costs)
        if costs[best_idx] < self.gbest_cost:
            self.gbest = self.particles[best_idx].pbest_position.copy()
            self.gbest_cost = costs[best_idx]

    async def optimize_async(self, objective_function,num_iterations):
        """
        Performs asynchronous optimization using PSO algorithm.

        This method initializes the particle swarm with the given objective function and
        performs optimization over a specified number of iterations. The inertia weight
        is adjusted linearly from its initial value to a minimum specified value throughout
        the iterations. The performance of the optimization process is tracked by recording
        the cost history after each update. The method performs all operations asynchronously
        to allow for non-blocking execution.

        Arguments:
            - objective_function (Callable): The function to be optimized. It should accept a
              position vector as input and return a scalar cost value.
            - num_iterations (int): The total number of iterations to be performed during the
              optimization process.

        Returns:
            tuple: A tuple containing:
                - gbest: The global best position found by the particle swarm.
                - gbest_cost: The global best cost associated with the best position.
                - cost_history: A list of cost values representing the global best cost after
                  each iteration.

        """
        await self.initialize_async(objective_function)
        w_min = .4
        w_values = np.linspace(self.w_init, w_min, num_iterations)
        cost_history = [self.gbest_cost]
        for w in w_values:
            await self.move_particles_async(w, objective_function)
            cost_history.append(self.gbest_cost)
        return self.gbest_cost, self.gbest, cost_history

async def _async_pso_runner(function,bounds,n_particles,iters,w,c1,c2,dim,vel_min,vel_max):
    """
    Runs PSO algorithm asynchronously on a given objective function to minimize or optimize the outcome.
    Utilizes the AsyncPSO optimizer to perform calculations across a specified number of particles and
    iterations, using provided parameters for the optimization process.

    Parameters:
        function (callable): The target objective function to optimize.
        bounds (tuple): A tuple containing two np.ndarrays  (lower bounds, upper bounds),
            specifying permissible ranges for each dimension of the optimization.
        n_particles (int): The number of particles in the swarm.
        iters (int): The number of optimization iterations to perform.
        w (float): The inertia coefficient, controlling the influence of a particle's previous velocity.
        c1 (float): The cognitive coefficient, governing the influence of a particle's personal best position.
        c2 (float): The social coefficient, governing the influence of the swarm's best-known position.
        dim (int): The number of dimensions for the optimization problem.
        vel_min (float): The minimum allowed velocity for particles across any dimension.
        vel_max (float): The maximum allowed velocity for particles across any dimension.

    Returns:
        tuple: The result of the optimization, as returned by the AsyncPSO optimizer.

    """
    def obj_function(x):
        return objective(position=x.reshape(1, -1), function=function)[0]
    optimizer = AsyncPSO(n_particles, dim, bounds, vel_min, vel_max, w, c1, c2)
    return await optimizer.optimize_async(obj_function, iters)

def my_async_pso(function, bounds, n_particles, iters, w, c1, c2, dim, vel_min, vel_max):
    """
    Executes AsyncPSO algorithm by creating an event loop and managing the lifecycle
    for an internal coroutine that performs the optimization process.

    Parameters:
        function (Callable): The objective function to optimize.
        bounds (Tuple[np.ndarray, np.ndarray]): Bounds for each dimension of the search
            space, where each tuple specifies the minimum and maximum values.
        n_particles (int): The number of particles in the swarm.
        iters (int): The number of iterations to run the optimization.
        w (float): The inertia coefficient affecting particles' velocity.
        c1 (float): The cognitive coefficient influencing personal best exploration.
        c2 (float): The social coefficient affecting swarm-wide exploration.
        dim (int): The dimensionality of the search space.
        vel_min (float): The minimum velocity a particle can have in any dimension.
        vel_max (float): The maximum velocity a particle can have in any dimension.

    Returns:
        Tuple[float,np.ndarray,list]: The result of the optimization process, as determined by the implementation
            of the `_async_pso_runner` coroutine.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            _async_pso_runner(function, bounds, n_particles, iters, w, c1, c2, dim, vel_min, vel_max)
        )
    finally:
        loop.close()
    return result