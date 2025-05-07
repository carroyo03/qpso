from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Optional, Dict, Any, Tuple, List

class ParticleInterface(ABC):
    """Interface for particle implementations across different backends."""
    
    @abstractmethod
    def update_velocity(self, global_best: np.ndarray, inertia: float, 
                        cognitive: float, social: float) -> None:
        """Update particle velocity."""
        pass
    
    @abstractmethod
    def update_position(self) -> None:
        """Update particle position."""
        pass
    
    @abstractmethod
    def evaluate(self, objective_function: Callable) -> float:
        """Evaluate particle fitness."""
        pass

class SwarmInterface(ABC):
    """Interface for swarm implementations across different backends."""
    
    @abstractmethod
    def initialize(self, n_particles: int, dimensions: int, 
                  bounds: Tuple[np.ndarray, np.ndarray]) -> None:
        """Initialize the swarm."""
        pass
    
    @abstractmethod
    def optimize(self, objective_function: Callable, max_iterations: int, 
                inertia: float, cognitive: float, social: float) -> Dict[str, Any]:
        """Run the optimization process."""
        pass
    
    @abstractmethod
    def get_best_solution(self) -> Tuple[np.ndarray, float]:
        """Return the best solution found."""
        pass
    
    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check if this backend is available on the current system."""
        pass

class BackendManager:
    """Manager for selecting and initializing appropriate backends."""
    
    @staticmethod
    def list_available_backends() -> List[str]:
        """List all available backends on the current system."""
        pass
    
    @staticmethod
    def get_backend(name: Optional[str] = None) -> SwarmInterface:
        """
        Get a specific backend by name or the best available backend.
        
        If name is None, automatically selects the most efficient backend
        available on the current system.
        """
        pass
    
    @staticmethod
    def benchmark_backends(objective_function: Callable, 
                          problem_size: int, 
                          n_particles: int,
                          max_iterations: int) -> Dict[str, Dict[str, Any]]:
        """
        Run a benchmark on all available backends and return performance metrics.
        """
        pass