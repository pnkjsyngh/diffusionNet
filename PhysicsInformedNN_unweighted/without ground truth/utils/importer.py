## Import the packages ##
import os
import numpy as np
import importlib
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import torch
from collections import OrderedDict
from scipy.interpolate import griddata
import pickle as pkl
from pyDOE import lhs

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Check if py-pde is installed ##
if importlib.util.find_spec('pde') is None:
    print('py-pde' +" is not installed")
    os.system('pip install py-pde')
    print('py-pde' +" is now installed")

from pde import PDE, CartesianGrid, ScalarField, MemoryStorage, plot_kymograph
from pde import Controller, ExplicitSolver, ScipySolver, ImplicitSolver

# ## Make sure change this on your end ##
# from google.colab import drive
# drive.mount('/content/drive')
# os.chdir('/content/drive/MyDrive/Colab Notebooks/PINNs/Diffusion')
# ## --------------------------------- ##
