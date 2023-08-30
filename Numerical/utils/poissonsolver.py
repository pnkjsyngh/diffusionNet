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
        ## Store the metadata ##
        self.metadata = {'field_str': field_str,
                         'edge_length': edge_length,
                         'ndiv': ndiv,
                         'bcs': bcs}
        
        ## Generate Grid ##
        self.grid = CartesianGrid([[0, edge_length]] * 2, ndiv)

        ## Generate field ##
        self.field = ScalarField.from_expression(self.grid, field_str)

        ## Assign boundary conditions ##
        self.bc = bcs
    
    ## Solve the PDE
    def solve(self):

        ## Solve the Poisson Equation ##
        self.result = solve_poisson_equation(self.field, self.bc)
    
    ## Get the value of solution at random points
    def get_val_random_points(self, n):
        
        ## Check if solution is generated or not ##
        if ~hasattr(self,'result'):
            self.solve()
            
        ## Make an empty array to store random points and the value at those points ##
        data = np.zeros(shape = (n,3))
        
        ## Run over the interator to get the coordinates ##
        for i in np.arange(n):
            data[i, :2] = self.grid.get_random_point(boundary_distance = self.metadata['edge_length']*0.05)
            data[i,-1] = self.result.interpolate(data[i, :2])
        
        return data           
    
    ## Get data along a slice ##
    def get_val_slice(self, axis, location):
        
        ## Extract slice data ##
        slice_data = self.result.slice({axis: location})
        
        ## Assign an array to store x,y and scalar value ##
        data = np.zeros(shape = (self.metadata['ndiv'],3))
        
        ## Fill the coordinates of the slice ##
        if axis == 'x':
            data[:,0] = np.ones(self.metadata['ndiv'])*location
            print(slice_data.grid._axes_coords[0])
            data[:,1] = slice_data.grid._axes_coords[0]
        else:
            data[:,0] = slice_data.grid._axes_coords[0]
            data[:,1] = np.ones(self.metadata['ndiv'])*location
        
        ## store the extracted field values in the array as well ##
        data[:,2] = slice_data.data
        
        return data