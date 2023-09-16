from utils.importer import *

## Initiate the class Poisson Solver ##
class NumPDE():

    ## Initiate the object with the input of the equation and variable to be solved ##
    def __init__(self,
                 eq,   # Provide the equation in the form of PDE object which contains equation and bcs
                 grid, # Provide the grid for the solution
                 field,# Provide the target variable field
                 solver: str,
                 trange: float,
                 dt: float):

        ## Store the input equation
        self.eq = eq

        ## Store the grid
        self.grid = grid

        ## Store the field variable
        self.field = field

        ## Store the metadata
        self.metadata = {'solver': solver, 'trange': trange, 'dt': dt}

        ## run the solver
        self.solvePDE()

        ## Organize the ground truth in torch compatible format
        self.get_ground_truth()
        
        ## Save the object into a pickle file
        with open('saved/numSol.pkl', 'wb') as file:        
            pkl.dump(self.ground_truth, file)


    ## Solve the PDE
    def solvePDE(self):

        ## Define time storage
        storage = MemoryStorage()

        ## Define solver
        if self.metadata['solver'] == 'Explicit':
            solver = ExplicitSolver(self.eq, scheme="runge-kutta", adaptive=True)
        elif self.metadata['solver'] == 'Scipy':
            solver = ScipySolver(eq)
        elif self.metadata['solver'] == 'Implicit':
            solver = ImplicitSolver(eq)
        else:
            print('Please provide a viable solver')

        ## Define the controller
        controller = Controller(solver, t_range=self.metadata['trange'],
                                tracker=storage.tracker(self.metadata['dt']))

        ## Run the solver
        solution = controller.run(self.field, dt=1e-3)

        ## Store the solution over each time step
        self.solution = storage

        # ## Show a solution contour
        # plot_kymograph(storage)

    ## Get the ground truth data
    def get_ground_truth(self):

        ## Time steps in the storage
        t = np.asarray(self.solution.times)

        ## Discretization along the space
        x = self.grid._axes_coords[0]

        ## Extract the solution
            ## Empty array to store the solution
        u = np.zeros(shape = (len(t), len(x)))
            ## Loop over storage
        i = 0
        for _, field in self.solution.items():
            u[i,:] = field.data
            i+=1

        ## Store the solution and grid points in dictionary
        self.ground_truth = {'x': x, 't': t, 'u': u}