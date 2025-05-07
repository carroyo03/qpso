import importlib
import inspect
import logging
import platform
import multiprocessing as mp
import subprocess
from mpi4py import MPI
import os
import sys
import time
import numpy as np
from typing import Dict, List, Optional, Type, Any, Tuple

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class BackendManager:

    _backends: Dict[str, Type] = {}

    _backend_capabilities: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_backend(cls, backend_name: str, backend_class: Type, capabilities: Dict[str, Any] = None) -> None:
        """
        Registers a new backend with the BackendManager.

        This method adds a new backend to the list of available backends,
        along with its corresponding capabilities. If the backend is already
        registered, it will be overwritten.

        Parameters:
        - backend_name (str): The unique name of the backend.
        - backend_class (Type): The class representing the backend.
        - capabilities (Dict[str, Any], optional): A dictionary containing the
        capabilities of the backend. Defaults to an empty dictionary.

        """
        cls._backends[backend_name] = backend_class
        cls._backend_capabilities[backend_name] = capabilities or {}
        logger.debug(f"Backend {backend_name} registered successfully")

    @classmethod
    def list_available_backends(cls) -> List[str]:
        """
        Returns a list of available backends based on their availability status.

        The method iterates through the registered backends and checks their availability
        using the `is_available` method. If a backend is available, its name is added to
        the `available` list.

        Returns:
        - List[str]: A list of available backend names.
        """
        available = []
        for backend_name, backend_class in cls._backends.items():
            if hasattr(backend_class, "is_available") and backend_class.is_available():
                available.append(backend_name)
        return available
    
    def get_backend(cls, name: Optional[str] = None) -> Type:
        """
        Retrieves an instance of a backend based on the provided name. If no name is provided,
        the most efficient available backend is selected.

        Parameters:
        - name (str, optional): The name of the backend. If not provided, the most efficient
          available backend will be selected. Defaults to None.

        Returns:
        - Type: An instance of the selected backend class.

        Raises:
        - RuntimeError: If no available backends are found.
        - ValueError: If an unsupported backend name is provided or if no class is found for the backend.
        - ValueError: If the selected backend is not available in the current environment.
        """

        available_backends = cls.list_available_backends()
        if not available_backends:
            raise RuntimeError("No available backends")
        if name is None:
            backend_priorities = {
                'openmp': 100,
                'metal': 90,
                'mpi': 80,
                'dask': 70,
                'numba': 60,
                'async_prog': 50,
                'threading': 40,
                'base': 10
            }

            sorted_backends = sorted(available_backends, key=lambda x: backend_priorities.get(x, 0), reverse=True)
            name = sorted_backends[0]
            logger.info(f"No backend specified. Using the most efficient available backend: {name}")
        elif name not in available_backends:
            raise ValueError(f"Unsupported backend: {name}")
        
        backend_class = cls._backends.get(name)
        if backend_class is None:
            raise ValueError(f"No class found for backend: {name}")
        
        if hasattr(backend_class, "is_available") and not backend_class.is_available():
            raise ValueError(f"Backend {name} is not available in this environment")

        return backend_class()
    
    @classmethod
    def get_backend_capabilities(cls, name: str) -> Dict[str, Any]:
        """
        Retrieves the capabilities of a specific backend based on its name.

        Parameters:
        - name (str): The name of the backend.

        Returns:
        - Dict[str, Any]: A dictionary containing the capabilities of the backend.
          If the backend is not found, an empty dictionary is returned.

        Raises:
        - ValueError: If an unsupported backend name is provided.
        """

        if name not in cls._backends:
            raise ValueError(f"Unsupported backend: {name}")
        return cls._backend_capabilities.get(name, {})
    
    @classmethod
    def benchmark_backends(cls, problem_size: int, n_particles: int, max_iterations: int, 
                    function:str, bounds: Tuple) -> Dict[str, Any]:
        results = {}
        available_backends = cls.list_available_backends()


        if bounds is None:
            bounds = (np.array([-5] * problem_size), np.array([5] * problem_size))
        
        for backend_name in available_backends:
            try:
                swarm = cls.get_backend(backend_name)

                if hasattr(swarm, "initialize"):
                    swarm.initialize(n_particles, problem_size, bounds)

                start_time = time.time()

                if hasattr(swarm, "optimize"):
                    optimization_result = swarm.optimize(function, max_iterations)
                
                end_time = time.time()

                results[backend_name] = {
                    'execution_time': end_time - start_time,
                    'best_cost': getattr(optimization_result, 'best_cost', float('inf')),
                    'iterations': max_iterations,
                    'n_particles': n_particles,
                    'problem_size': problem_size
                }

                logger.info(f'Benchmark for {backend_name} completed successfully in {results[backend_name]["execution_time"]} seconds')
            except Exception as e:
                logger.error(f"Error benchmarking {backend_name}: {str(e)}")
                results[backend_name] = {
                    'error': str(e)
                }
            
        return results


    @classmethod
    def detect_hardware(cls) -> Dict[str, Any]:
        """
        Detects the hardware specifications of the current system.

        This function retrieves various hardware specifications such as the platform, processor, 
        CPU count, Python version, MAC processor (if available), and MPI information (if available).

        Returns:
        Dict[str, Any]: A dictionary containing the detected hardware specifications.
        """

        hardware_info = {
            'platform': platform.system(),
            'processor': platform.processor(),
            'cpu_count': mp.cpu_count(),
            'python_version' : platform.python_version()
        }

        if platform.system() == 'Darwin':
                try:
                    result = subprocess.run(
                        ['sysctl', '-n', 'machdep.cpu.brand_string'],
                        capture_output=True,
                        text=True
                    )

                    hardware_info['mac_processor'] = result.stdout.strip()

                    hardware_info['is_apple_silicon'] = 'Apple' in hardware_info['mac_processor']
                except Exception:
                    hardware_info['is_apple_silicon'] = False
                
        try:
            hardware_info['mpi_available'] = True
            hardware_info['mpi_size'] = MPI.COMM_WORLD.Get_size()
            hardware_info['mpi_rank'] = MPI.COMM_WORLD.Get_rank()
        except Exception:
            hardware_info['mpi_available'] = False
                
        return hardware_info

    @classmethod
    def recommend_backend(cls) -> str:
        """
        Recommends the most suitable backend based on the detected hardware specifications and available backends.

        The function first retrieves hardware specifications using the `detect_hardware` method. It then retrieves
        a list of available backends using the `list_available_backends` method.

        If no available backends are found, the function returns 'base'.

        If the hardware is an Apple Silicon device and the 'metal' backend is available, the function returns 'metal'.

        If MPI is available and the MPI size is greater than 1, and the 'mpi' backend is available, the function returns 'mpi'.

        If the CPU count is greater than 1, the function checks for the availability of the 'openmp' and 'threading' backends.
        If 'openmp' is available, it returns 'openmp'. If 'threading' is available, it returns 'threading'.

        If none of the above conditions are met, the function returns the first available backend.

        Returns:
        str: The recommended backend name.
        """

        hardware_info = cls.detect_hardware()
        available_backends = cls.list_available_backends()

        if not available_backends:
            return 'base'
        
        if hardware_info.get('is_apple_silicon', False) and 'metal' in available_backends:
            return 'metal'
        elif hardware_info.get('mpi_available', False) and hardware_info.get('mpi_size', 1) > 1 and 'mpi' in available_backends:
            return 'mpi'
        elif hardware_info.get('cpu_count', 1) > 1:
            if 'openmp' in available_backends:
                return 'openmp'
            elif 'threading' in available_backends:
                return 'threading'
        
        return available_backends[0]

    @classmethod
    def discover_backends(cls) -> None:
        """
        Discovers and registers available Particle Swarm Optimization (PSO) backends.

        The function searches for PSO backend implementations in the 'implementations' directory.
        It iterates through each subdirectory in the 'implementations' directory, checks for the existence
        of a 'pso' directory, and then attempts to import the 'pso' module.

        If the 'pso' module is successfully imported, the function iterates through the module's members.
        It identifies classes that represent PSO backends (excluding 'SwarmInterface' and 'Swarm') and checks
        if the backend has an 'is_available' method. If the backend is available, it is registered with the
        BackendManager using the backend's directory name as the backend name.

        If an ImportError or any other exception occurs during the discovery process, the function logs the error
        using the logger.

        Args:
        None

        Returns:
        None
        """

        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        implementations_path = os.path.join(base_path, "implementations")
        
        if not os.path.exists(implementations_path):
            logger.warning(f"Implementations path not found: {implementations_path}")
            return
        
        for implementation_dir in os.listdir(implementations_path):
            implementation_path = os.path.join(implementations_path, implementation_dir)

            if not os.path.isdir(implementation_path):
                continue

            pso_path = os.path.join(implementation_path, "pso")
            if not os.path.exists(pso_path) or not os.path.isdir(pso_path):
                continue

            module_path = os.path.join(pso_path, "pso.py")

            if not os.path.exists(module_path):
                continue

            try:

                import_path = f"implementations.{implementation_dir}.pso.pso"
                module = importlib.import_module(import_path)

                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and name not in ['SwarmInterface', 'Swarm']:
                        if hasattr(obj, 'is_available'):
                            cls.register_backend(implementation_dir, obj)
                            logger.info(f"Backend {implementation_dir} found & registered successfully")
                            break
            except ImportError as e:
                logger.error(f"Error importing backend {implementation_dir}: {str(e)}")
            except Exception as e:
                logger.error(f"Error finding backend {implementation_dir}: {str(e)}")
                
"""
# Inicializar el gestor de backends
BackendManager.discover_backends()

# Listar backends disponibles
available_backends = BackendManager.list_available_backends()
print(f"Backends disponibles: {available_backends}")

# Obtener el backend recomendado
recommended = BackendManager.recommend_backend()
print(f"Backend recomendado: {recommended}")

# Obtener una instancia del backend
swarm = BackendManager.get_backend("openmp")

# Ejecutar benchmark
results = BackendManager.benchmark_backends(
    problem_size=10,
    n_particles=30,
    max_iterations=100,
    function="rosenbrock"
)

# Mostrar resultados del benchmark
for backend, metrics in results.items():
    print(f"Backend: {backend}")
    print(f"Tiempo: {metrics['execution_time']:.4f} segundos")
    print(f"Mejor costo: {metrics['best_cost']}")

"""