## $${\color{blue}\mathbf{Solution \space of \space 1D \space transient \space diffusion \space PDE \space using \space Neural \space Networks}}$$ ##

### $${\color{red}\mathbf{Problem \space defintion}}$$ ###

---

The governing PDE for 1D diffusion is given by,

$$\\frac{\\partial T}{\\partial t} = \\frac{\\partial^2 T}{\\partial x^2} + s(t) $$ 

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


### $${\color{red}\mathbf{Data \space based \space Neural \space Networks}}$$ ###
