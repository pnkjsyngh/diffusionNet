## Solution of 1D transient diffusion PDE using Neural Networks

### Problem Defintion ##
The governing PDE for homogeneous 1D diffusion is given by,

$$\\frac{\\partial T}{\\partial t} = \\frac{\\partial^2 T}{\\partial x^2}$$ 

This PDE is valid for,

$$\\forall \\, x \\in (0,l) \\text{ and } 0 < t < t_{range}$$

The initial condition for this PDE is, 

$$T(x, 0) = 0 $$

The boundary conditions for this PDE are,

$$ \\left.\\frac{\\partial T}{\\partial x}\\right\\vert_{(0,t)} = 0 \\text{ and } \\left. T \\right\\vert_{(l,t)} = sin(t)$$
