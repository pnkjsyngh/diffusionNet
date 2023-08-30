## Import basic packages ##
import matplotlib.pyplot as plt
import numpy as np
import os
import importlib.util


## Check if py-pde is installed ##
if importlib.util.find_spec('pde') is None:
    print('py-pde' +" is not installed")
    os.system('pip install py-pde')
    print('py-pde' +" is now installed")

## When py-pde is found import packages ##
from pde import CartesianGrid, solve_poisson_equation, ScalarField

## Initiate the class Poisson Solver ##
class PoissonSolver():

  def __init__(self,
               field_str: str,
               edge_length: float,
               ndiv: int,
               bcs: np.ndarray):

    ## Generate Grid ##
    self.grid = CartesianGrid([[0, edge_length]] * 2, ndiv)

    ## Generate field ##
    self.field = ScalarField.from_expression(self.grid, field_str)

    ## Assign boundary conditions ##
    self.bc = bcs

  def solve(self):

    ## Solve the Poisson Equation ##
    self.result = solve_poisson_equation(self.field, self.bc)