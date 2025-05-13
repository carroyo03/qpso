import numpy as np
from numba import njit

@njit(parallel=True)
def compute_velocity(velocity, position, pbest_position, gbest_position, w, c1, c2, vel_min, vel_max):
    """
    Compute the updated velocity for a particle based on its current
    velocity, position, personal best position, and global best position,
    along with given parameters for inertia weight, cognitive
    coefficient, social coefficient, and velocity bounds.

    Args:
        velocity (numpy.ndarray): The current velocity of the particle.
        position (numpy.ndarray): The current position of the particle.
        pbest_position (numpy.ndarray): The personal best position of
            the particle.
        gbest_position (numpy.ndarray): The global best position in the
            swarm.
        w (float): The inertia weight controlling the impact of the
            current velocity.
        c1 (float): The cognitive coefficient controlling the
            attraction toward the personal best.
        c2 (float): The social coefficient controlling the attraction
            toward the global best.
        vel_min (float): The lower boundary for velocity values.
        vel_max (float): The upper boundary for velocity values.

    Returns:
        numpy.ndarray: The updated velocity of the particle.

    Raises:
        None
    """
    r1 = np.random.random(size=velocity.shape)
    r2 = np.random.random(size=velocity.shape)
    cognitive = c1 * r1 * (pbest_position - position)
    social = c2 * r2 * (gbest_position - position)
    new_velocity = w * velocity + cognitive + social
    return np.clip(new_velocity, vel_min, vel_max)

@njit(parallel=True)
def compute_position(position, velocity, pos_min, pos_max):
    """
    Computes the updated position by adding the velocity to the current position and ensures the result is
    clamped within the provided minimum and maximum positional bounds using NumPy's clip function.

    Parameters:
    position : array-like or float
        The current position value(s).
    velocity : array-like or float
        The velocity value(s) to be added to the current position.
    pos_min : array-like or float
        The minimum positional bound(s).
    pos_max : array-like or float
        The maximum positional bound(s).

    Returns:
    array-like or float
        The updated position value(s), clamped between `pos_min` and `pos_max`.
    """
    new_position = position + velocity
    return np.clip(new_position, pos_min.astype(np.float64), pos_max.astype(np.float64))

class Particle:

    def __init__(self, dim: int, pos_min: np.ndarray, pos_max: np.ndarray,
                vel_min: float, vel_max: float):
        """
        The function initializes a particle with random position and velocity within specified limits
        for a particle swarm optimization algorithm.
        
        Args:
          dim (int): The `dim` parameter in the `__init__` method represents the dimensionality of the
        problem space. It indicates the number of dimensions in which the particles in the optimization
        algorithm will be moving and searching for the optimal solution.
          pos_min (np.ndarray): `pos_min` is a numpy array representing the minimum position values for
        each dimension of the particle in a particle swarm optimization algorithm. It defines the lower
        bounds within which the particle's position can vary during optimization.
          pos_max (np.ndarray): The `pos_max` parameter represents the maximum position values allowed
        for each dimension in the search space. It is used to define the upper boundary for the randomly
        initialized position of a particle in a particle swarm optimization algorithm. The particle's
        position is initialized within the range defined by `pos_min` and `
          vel_min (float): The `vel_min` parameter in the `__init__` method represents the minimum
        velocity value that can be assigned to the particle during initialization. This value is used to
        set the lower bound for the random initialization of the velocity components of the particle
        within a certain range. It helps control the speed at
          vel_max (float): The `vel_max` parameter in the `__init__` method appears to represent the
        maximum velocity value allowed for the particle in a particle swarm optimization (PSO)
        algorithm. This value is used to initialize the velocity of the particle within a certain range
        during the initialization phase.
        """
       
        
        # Inicializa la posición aleatoriamente dentro de los límites
        self.position = np.random.uniform(low=pos_min, high=pos_max, size=dim)
        
        # Inicializa la velocidad aleatoriamente (típicamente menor que el rango de posición)
        self.velocity = np.random.uniform(
            low=-0.1 * (pos_max - pos_min),
            high=0.1 * (pos_max - pos_min),
            size=dim
        )
        
        # Inicializa el mejor personal
        self.pbest_position = self.position.copy()
        self.pbest_cost = float('inf')  # Se actualizará en la primera evaluación
        
        # Almacena los límites para recorte
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.vel_min = vel_min
        self.vel_max = vel_max

    def update_velocity_and_position(self, w: float, c1: float, c2: float, gbest_position: np.ndarray):
        """
        Updates the velocity and position of the particle based on the provided
        coefficients and the global best position.

        The method incorporates the effects of inertia, cognitive behavior, and
        social behavior to calculate the new velocity, then updates the position
        accordingly within defined limits.

        Parameters:
        w: float
            The inertia weight factor, controlling the influence of the previous
            velocity.
        c1: float
            The cognitive coefficient, determining how much the particle is
            influenced by its best-known position.
        c2: float
            The social coefficient, dictating the particle's influence by the
            swarm's best-known position.
        gbest_position: np.ndarray
            The global best-known position in the swarm, used for the social
            behavior component.


        """
        self.velocity = compute_velocity(
            self.velocity, self.position, self.pbest_position, gbest_position,
            w, c1, c2,self.vel_min, self.vel_max
        )
        self.position = compute_position(
            self.position, self.velocity, self.pos_min, self.pos_max
        )


    def evaluate(self, objective_function):
        """
        This function evaluates the current position, updates the personal best if the current position
        is better, and returns the cost.
        
        Args:
          objective_function: The `objective_function` parameter in the `evaluate` method is a function
        that takes a position as input and returns a cost value. This function is used to evaluate the
        current position of a particle in a particle swarm optimization algorithm. The cost value
        represents how well the particle's position performs with respect to
        
        Returns:
          The `evaluate` method returns the cost of the current position after evaluating it with the
        objective function.
        """
       
        
        # Evalúa la posición actual
        cost = objective_function(self.position)
        
        # Actualiza el mejor personal si la posición actual es mejor
        if cost < self.pbest_cost:
            self.pbest_position = self.position.copy()
            self.pbest_cost = cost
            
        return cost
    
    def __del__(self):
      """
      The `__del__` method is a special method that is automatically called when an object is deleted
      from memory. In this case, it is used to release any resources or memory allocated to the
      particle object.
      """
      pass