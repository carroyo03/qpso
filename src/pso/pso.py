import numpy as np
from tqdm import tqdm #type: ignore


class ParticleSwarmOptimization:
    def __init__(self, number_of_particles: int, dim: int,
                 bounds: tuple, vel_min: float, vel_max: float,
                 w: float, c1: float, c2: float, function: str):
        """ParticleSwarmOptimization (PSO) algorithm implementation.
        Args:
            number_of_particles (int): Number of particles in the swarm.
            dim (int): Number of dimensions for the optimization problem.
            bounds (tuple): Tuple containing the minimum and maximum bounds for the position.
            vel_min (float): Minimum velocity for the particles.
            vel_max (float): Maximum velocity for the particles.
            w (float): Inertia weight.
            c1 (float): Cognitive coefficient.
            c2 (float): Social coefficient.
            function (str): The objective function to optimize. Supported functions: "ackley", "rastrigin", "rosenbrock".
        """
        pos_min, pos_max = bounds
        pos_min = np.array(pos_min)
        pos_max = np.array(pos_max)

        if np.any(pos_min >= pos_max):
            raise ValueError("The minimum position cannot be greater or equal than the maximum")

        self.number_of_particles = number_of_particles
        self.dim = dim
        self.pos_min, self.pos_max = pos_min, pos_max
        self.vel_min, self.vel_max = vel_min, vel_max
        self.w_init = w  # Initial inertia weight
        self.c1, self.c2 = c1, c2
        self.function = function

        # Initialize particles' positions and velocities
        self.positions = np.random.uniform(low=pos_min, high=pos_max, size=(number_of_particles, dim))
        self.velocities = np.random.uniform(low=-0.1 * (pos_max - pos_min),
                                            high=0.1 * (pos_max - pos_min),
                                            size=(number_of_particles, dim))

        # Initialize personal best (pbest) and global best (gbest)
        self.pbest = self.positions.copy()
        self.pbest_costs = self.objective_function(self.positions)
        gbest_idx = np.argmin(self.pbest_costs)
        self.gbest = self.pbest[gbest_idx].copy()
        self.gbest_cost = self.pbest_costs[gbest_idx]

        # Initialize cost history
        self.cost_history = []

    def objective_function(self, position:np.ndarray) -> np.ndarray:
        """Evaluate the objective function.
        Args:
            position (ndarray): Particle positions.
        Returns:
            ndarray: Evaluated costs for each particle.
        """

        # Check if the function is supported
        if self.function not in ["ackley", "rastrigin", "rosenbrock"]:
            raise ValueError(f"Unsupported function: {self.function}")

        # Ensure position is not unidimensional
        if position.ndim == 1:
            position = position.reshape(1, -1)

        if self.function == "ackley":
            # Ackley function scenario
            a, b, c = 20, 0.2, 2 * np.pi
            sum1 = np.sum(position ** 2, axis=1)
            sum2 = np.sum(np.cos(c * position), axis=1)
            term1 = -a * np.exp(-b * np.sqrt(sum1 / self.dim))
            term2 = -np.exp(sum2 / self.dim)
            return term1 + term2 + a + np.e

        elif self.function == "rastrigin":
            # Rastrigin function scenario
            position = np.clip(position, -5.12, 5.12)
            return 10 * self.dim + np.sum(position ** 2 - 10 * np.cos(2 * np.pi * position), axis=1)

        else:
            # Rosenbrock function scenario
            result = np.zeros(position.shape[0])
            for i in range(self.dim - 1):
                x = position[:, i]
                y = position[:, i + 1]
                result += 100 * (y - x ** 2) ** 2 + (1 - x) ** 2
            return result

    def move_particles(self, w: float):
        """Update the particles' positions and velocities.
        Args:
            w (float): Inertia weight.
        """
        # Generate random numbers for cognitive and social components
        r1 = np.random.rand(self.number_of_particles)
        r2 = np.random.rand(self.number_of_particles)

        # Update velocities
        cognitive = self.c1 * r1[:, np.newaxis] * (self.pbest - self.positions)
        social = self.c2 * r2[:, np.newaxis] * (self.gbest - self.positions)
        self.velocities = w * self.velocities + cognitive + social
        self.velocities = np.clip(self.velocities, self.vel_min, self.vel_max)

        # Update positions
        self.positions += self.velocities
        self.positions = np.clip(self.positions, self.pos_min, self.pos_max)

        # Update pbest and gbest
        costs = self.objective_function(self.positions)
        improved = costs < self.pbest_costs
        self.pbest[improved] = self.positions[improved].copy()
        self.pbest_costs[improved] = costs[improved]

        # Update gbest if a better pbest is found
        gbest_idx = np.argmin(self.pbest_costs)
        if self.pbest_costs[gbest_idx] < self.gbest_cost:
            self.gbest = self.pbest[gbest_idx].copy()
            self.gbest_cost = self.pbest_costs[gbest_idx]

    def optimize(self, num_iterations: int) -> tuple:
        """Optimize the objective function using PSO.
        Args:
            num_iterations (int): Number of iterations for the optimization.
        Returns:
            tuple: Best cost, best position, and cost history.
        """
        # Check if gbest is initialized
        if self.gbest is None:
            raise ValueError("self.gbest is not initialized")

        # Initialize the inertia weight schedule
        # Linearly decrease the inertia weight from w_init to w_min
        w_min = 0.4
        w_values = np.linspace(self.w_init, w_min, num_iterations)

        # Initialize cost history
        self.cost_history = []
        
        # Add progress bar for iterations
        for it in tqdm(range(num_iterations), desc="PSO Progress", leave=False):
            # Update the inertia weight and move particles
            self.move_particles(w_values[it])
            self.cost_history.append(self.gbest_cost)

        # Final evaluation
        return self.gbest_cost, self.gbest, self.cost_history