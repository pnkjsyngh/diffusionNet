## $${\color{blue}\mathbf{Solution \space of \space 1D \space transient \space diffusion \space PDE \space using \space Neural \space Networks}}$$ ##

### $${\color{red}\mathbf{Problem \space defintion}}$$ ###

The governing PDE for 1D diffusion is given by,

$$\\frac{\\partial T}{\\partial t} = D \\frac{\\partial^2 T}{\\partial x^2} + s(x,t)$$ 

This PDE is valid for,

$$\\forall \\, x \\in (0,l) \\text{ and } 0 < t < t_{range}$$

The initial condition for this PDE is, 

$$T(x, 0) = i(x) $$

The boundary conditions for this PDE are,

$$ \\left.\\frac{\\partial T}{\\partial x}\\right\\vert_{(0,t)} = a(x, t) \\text{ and } \\left. T \\right\\vert_{(l,t)} = b(x, t)$$

where,<br>
$s(t)$ is the source term,<br>
$i(x)$ is the initial temperature distribution on the domain,<br>
$a(t)$ and $b(t)$ are time dependent boundary conditions terms at $x=0$ and $l$ respectively.
<br>

### $${\color{red}\mathbf{Numerical \space solution}}$$ ###

The numerical solution for this problem is obtained using [py-pde](https://py-pde.readthedocs.io/en/latest/) for following set of system parameters,
- $D = 0.1$
- $l = 1$
- $t_{range} = 2\pi$

The specified functions are as follows,
- $s(x, t) = 0$
- $i(x) = 0$
- $a(t) = 0$
- $b(t) = sin(t)$

<p align="center">
  <img src="./PINN_unweighted_loss/only%20physics/results/ground_truth.png", width=400px>
</p>

### $${\color{red}\mathbf{\text{Physics Informed Neural Networks (PINNs) with unweighted loss function}}}$$ ###

We define the fully physics based loss function as follows for training our neural network,

$$\mathcal{L} = \mathcal{L_{IC}} + \mathcal{L_{BC}} + \mathcal{L_{PDE}}$$

where,<br>
$\mathcal{L}$ is the overall loss,<br>
$\mathcal{L_{IC}}$ is the loss from initial conditions at $t=0$ over the domain,<br>
$\mathcal{L_{BC}}$ is the loss from boundary conditions at boundaries ($x=0$ and $1$) over time,<br>
$\mathcal{L_{PDE}}$ is the loss from PDE at collocation points.<br>

As it can be noted, we have presented the loss as simple sum of losses coming from IC, BC and PDE. The implementation can be found in the [PINNs unweighted loss](./PhysicsInformedNN_unweighted/only%20physics) folder.

<p align="center">
  <img src="./PINN_unweighted_loss/only%20physics/results/losses.png", width=400px>
</p>

<p align="center">
  <img src="./PINN_unweighted_loss/only%20physics/results/contours.png">
</p>

<p align="center">
  <img src="./PINN_unweighted_loss/only%20physics/results/slices.png">
</p>
