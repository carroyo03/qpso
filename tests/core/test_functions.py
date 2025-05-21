import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from src.core.functions import ackley, rastrigin, rosenbrock, objective

# Test points
# For direct function calls, pass 2D arrays as functions expect (num_particles, dim)
zeros_1x2 = np.array([[0, 0]])
ones_1x2 = np.array([[1, 1]])
twos_1x2 = np.array([[2, 2]]) # Specifically for Rosenbrock

# For objective function wrapper, 1D arrays are fine as it reshapes
zeros_2d_objective = np.array([0, 0])
ones_2d_objective = np.array([1, 1])


# Expected values (generally 0 at the origin for Ackley and Rastrigin)
ACKLEY_AT_ZEROS = 0.0
RASTRIGIN_AT_ZEROS = 0.0
ROSENBROCK_AT_ONES = 0.0 # Global minimum

# Ackley function tests
def test_ackley_zeros():
    assert_almost_equal(ackley(zeros_1x2)[0], ACKLEY_AT_ZEROS, decimal=5)

def test_ackley_ones():
    expected = -20 * np.exp(-0.2 * np.sqrt(0.5 * (1**2 + 1**2))) - np.exp(0.5 * (np.cos(2*np.pi*1) + np.cos(2*np.pi*1))) + 20 + np.exp(1)
    assert_almost_equal(ackley(ones_1x2)[0], expected, decimal=5)

# Rastrigin function tests
def test_rastrigin_zeros():
    assert_almost_equal(rastrigin(zeros_1x2)[0], RASTRIGIN_AT_ZEROS, decimal=5)

def test_rastrigin_ones():
    # For 2D: 10*2 + (1^2 - 10*cos(2*pi*1)) + (1^2 - 10*cos(2*pi*1)) = 2
    expected = 2.0
    assert_almost_equal(rastrigin(ones_1x2)[0], expected, decimal=5)

# Rosenbrock function tests
def test_rosenbrock_zeros():
    # For 2D [[0,0]]: (0-1)^2 + 100(0-0^2)^2 = 1
    assert_almost_equal(rosenbrock(zeros_1x2)[0], 1.0, decimal=5)

def test_rosenbrock_ones():
    assert_almost_equal(rosenbrock(ones_1x2)[0], ROSENBROCK_AT_ONES, decimal=5)

def test_rosenbrock_twos():
    # For [[2,2]]: (2-1)^2 + 100(2-2^2)^2 = 1^2 + 100(2-4)^2 = 1 + 100(-2)^2 = 1 + 400 = 401
    assert_almost_equal(rosenbrock(twos_1x2)[0], 401.0, decimal=5)

# Objective wrapper function tests
def test_objective_ackley():
    assert_almost_equal(objective(zeros_2d_objective, 'ackley')[0], ACKLEY_AT_ZEROS, decimal=5)

def test_objective_rastrigin():
    assert_almost_equal(objective(zeros_2d_objective, 'rastrigin')[0], RASTRIGIN_AT_ZEROS, decimal=5)

def test_objective_rosenbrock():
    assert_almost_equal(objective(ones_2d_objective, 'rosenbrock')[0], ROSENBROCK_AT_ONES, decimal=5) # at (1,1) is 0
    assert_almost_equal(objective(zeros_2d_objective, 'rosenbrock')[0], 1.0, decimal=5) # at (0,0) is 1

def test_objective_unsupported_function():
    with pytest.raises(ValueError):
        objective(zeros_2d_objective, 'nonexistent_function')

def test_objective_1d_input_ackley():
    zeros_1d = np.array([0])
    assert_almost_equal(objective(zeros_1d, 'ackley')[0], 0.0, decimal=5)

def test_objective_1d_input_rastrigin():
    zeros_1d = np.array([0])
    assert_almost_equal(objective(zeros_1d, 'rastrigin')[0], 0.0, decimal=5)
    
def test_objective_1d_input_rosenbrock_error(): # Name kept for consistency, but it should pass now
    zeros_1d = np.array([0])
    ones_1d = np.array([1])
    # Rosenbrock for 1D: (x1-1)^2. For x1=0, (0-1)^2=1. For x1=1, (1-1)^2=0.
    assert_almost_equal(objective(zeros_1d, 'rosenbrock')[0], 1.0, decimal=5) 
    assert_almost_equal(objective(ones_1d, 'rosenbrock')[0], 0.0, decimal=5)

def test_functions_with_higher_dimensions():
    zeros_1x5 = np.array([[0, 0, 0, 0, 0]])
    ones_1x5 = np.array([[1, 1, 1, 1, 1]])
    
    # Ackley
    assert_almost_equal(ackley(zeros_1x5)[0], ACKLEY_AT_ZEROS, decimal=5)
    # Rastrigin
    assert_almost_equal(rastrigin(zeros_1x5)[0], RASTRIGIN_AT_ZEROS, decimal=5)
    # Rosenbrock
    # For [[0,0,0,0,0]]: (0-1)^2 * 4 = 4 (since all x_i are 0)
    assert_almost_equal(rosenbrock(zeros_1x5)[0], 4.0, decimal=5) 
    assert_almost_equal(rosenbrock(ones_1x5)[0], ROSENBROCK_AT_ONES, decimal=5)

# It's good practice to test edge cases or different configurations if applicable
# For these functions, the main variation is dimensionality, which is implicitly handled by numpy ops.
# Values at specific known points (like global minimums) are good checks.
