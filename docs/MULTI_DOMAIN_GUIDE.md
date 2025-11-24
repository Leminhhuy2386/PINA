# Multi-Domain Problem Solving with PINA

## Overview

**PINA fully supports multi-domain problems**, allowing you to solve complex problems where:

- Different physical laws apply in different regions of space
- Multiple materials or media are present with different properties
- Boundary conditions connect multiple domains
- Solutions must satisfy continuity conditions at domain interfaces

This guide explains how to leverage PINA's multi-domain capabilities for your scientific machine learning tasks.

## What are Multi-Domain Problems?

Multi-domain problems are common in physics and engineering. Examples include:

- **Heat conduction in composite materials**: Different thermal conductivities in different regions
- **Fluid-structure interaction**: Different governing equations for fluid and solid domains
- **Electromagnetic problems**: Different permittivity and permeability in different materials
- **Geophysical modeling**: Layered earth models with different rock properties
- **Biological systems**: Different tissue types with varying properties

## Core Multi-Domain Features in PINA

### 1. Multiple Domain Definitions

PINA allows you to define as many domains as needed using the `domains` dictionary in your problem class:

```python
from pina.problem import SpatialProblem
from pina.domain import CartesianDomain

class MultiDomainProblem(SpatialProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 2], "y": [0, 1]})
    
    domains = {
        "domain1": CartesianDomain({"x": [0, 1], "y": [0, 1]}),
        "domain2": CartesianDomain({"x": [1, 2], "y": [0, 1]}),
        "interface": CartesianDomain({"x": 1.0, "y": [0, 1]}),
        "boundary_left": CartesianDomain({"x": 0.0, "y": [0, 1]}),
        "boundary_right": CartesianDomain({"x": 2.0, "y": [0, 1]}),
    }
```

### 2. Domain-Specific Conditions

Apply different equations and boundary conditions to each domain:

```python
from pina import Condition
from pina.equation import Equation, FixedValue

conditions = {
    "physics_d1": Condition(domain="domain1", equation=Equation(equation1)),
    "physics_d2": Condition(domain="domain2", equation=Equation(equation2)),
    "bc_left": Condition(domain="boundary_left", equation=FixedValue(0.0)),
    "bc_right": Condition(domain="boundary_right", equation=FixedValue(1.0)),
}
```

### 3. Domain Operations

PINA provides powerful domain operations to create complex geometries:

- **Union**: Combine multiple domains
- **Intersection**: Find overlapping regions
- **Difference**: Subtract one domain from another
- **Exclusion**: Exclude specific regions

```python
from pina.domain import Union, Difference, Intersection, EllipsoidDomain

# Create a square with a circular hole
square = CartesianDomain({"x": [-1, 1], "y": [-1, 1]})
circle = EllipsoidDomain({"x": [-0.3, 0.3], "y": [-0.3, 0.3]})
domain_with_hole = Difference([square, circle])

# Union of multiple regions
domain1 = CartesianDomain({"x": [0, 1], "y": [0, 1]})
domain2 = CartesianDomain({"x": [1, 2], "y": [0, 1]})
combined_domain = Union([domain1, domain2])
```

### 4. Independent Domain Discretization

Sample different numbers of points in different domains:

```python
# Dense sampling in critical regions
problem.discretise_domain(n=500, mode="random", domains=["domain1"])

# Coarser sampling in less critical regions
problem.discretise_domain(n=100, mode="random", domains=["domain2"])

# Specific sampling at interfaces
problem.discretise_domain(n=50, mode="random", domains=["interface"])
```

### 5. Interface Domains

Define interface domains to handle continuity conditions between subdomains:

```python
domains = {
    "material1": CartesianDomain({"x": [0, 1], "y": [0, 1]}),
    "material2": CartesianDomain({"x": [1, 2], "y": [0, 1]}),
    "interface": CartesianDomain({"x": 1.0, "y": [0, 1]}),  # Interface at x=1
}
```

## Complete Example: Heat Conduction in Composite Material

Here's a complete example showing how to solve a heat conduction problem in a composite material with two different thermal conductivities:

```python
import torch
from pina import Trainer, Condition
from pina.problem import SpatialProblem
from pina.operator import laplacian
from pina.solver import PINN
from pina.model import FeedForward
from pina.domain import CartesianDomain
from pina.equation import Equation, FixedValue


def heat_equation_k1(input_, output_):
    """Heat equation for material 1 (k=1.0)"""
    k1 = 1.0
    laplace_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
    return k1 * laplace_u


def heat_equation_k2(input_, output_):
    """Heat equation for material 2 (k=0.1)"""
    k2 = 0.1
    laplace_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
    return k2 * laplace_u


class CompositeHeatProblem(SpatialProblem):
    """Multi-domain heat conduction problem"""
    
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 2], "y": [0, 1]})
    
    domains = {
        "material1": CartesianDomain({"x": [0, 1], "y": [0, 1]}),
        "material2": CartesianDomain({"x": [1, 2], "y": [0, 1]}),
        "left_boundary": CartesianDomain({"x": 0.0, "y": [0, 1]}),
        "right_boundary": CartesianDomain({"x": 2.0, "y": [0, 1]}),
    }
    
    conditions = {
        "physics_mat1": Condition(
            domain="material1", equation=Equation(heat_equation_k1)
        ),
        "physics_mat2": Condition(
            domain="material2", equation=Equation(heat_equation_k2)
        ),
        "bc_hot": Condition(domain="left_boundary", equation=FixedValue(1.0)),
        "bc_cold": Condition(domain="right_boundary", equation=FixedValue(0.0)),
    }


# Create and solve the problem
problem = CompositeHeatProblem()
problem.discretise_domain(n=300, mode="random", domains=["material1", "material2"])
problem.discretise_domain(n=30, mode="random", domains=["left_boundary", "right_boundary"])

model = FeedForward(
    layers=[40, 40, 40],
    output_dimensions=1,
    input_dimensions=2,
)

solver = PINN(problem, model)
trainer = Trainer(solver, max_epochs=1000, accelerator="cpu")
trainer.train()
```

## Key Advantages of PINA's Multi-Domain Support

1. **Flexibility**: Define as many domains as needed with different geometries
2. **Modularity**: Apply different physics/equations to different domains independently
3. **Seamless Integration**: Works with all PINA solvers (PINN, supervised, etc.)
4. **Domain Operations**: Create complex geometries using simple building blocks
5. **Adaptive Sampling**: Control point distribution in each domain separately
6. **Interface Handling**: Explicitly define and sample interface regions

## Common Use Cases

### 1. Composite Materials
```python
# Different material properties in different regions
domains = {
    "aluminum": CartesianDomain({...}),
    "steel": CartesianDomain({...}),
    "interface": CartesianDomain({...}),
}
```

### 2. Multi-Physics Coupling
```python
# Different physics in different domains
conditions = {
    "fluid_physics": Condition(domain="fluid_region", equation=navier_stokes),
    "solid_physics": Condition(domain="solid_region", equation=elasticity),
}
```

### 3. Complex Boundaries
```python
# Multiple boundary segments with different conditions
domains = {
    "interior": CartesianDomain({...}),
    "inlet": CartesianDomain({...}),
    "outlet": CartesianDomain({...}),
    "walls": Union([top_wall, bottom_wall, side_walls]),
}
```

### 4. Layered Systems
```python
# Stratified media
domains = {
    "layer1": CartesianDomain({"z": [0, 1]}),
    "layer2": CartesianDomain({"z": [1, 2]}),
    "layer3": CartesianDomain({"z": [2, 3]}),
}
```

## Best Practices

1. **Interface Sampling**: Always sample points at domain interfaces to ensure continuity
2. **Adaptive Resolution**: Use finer sampling in domains with rapid variations
3. **Domain Naming**: Use descriptive names for domains to improve code readability
4. **Visualization**: Plot sampled points to verify domain coverage before training
5. **Modular Design**: Keep domain and condition definitions clean and organized

## Examples and Tutorials

For detailed examples, see:

- **Tutorial 24**: [Multi-Domain Problem Solving](../tutorials/tutorial24/tutorial.py)
- **Tutorial 1**: [Simple ODE with multiple domains](../tutorials/tutorial1/tutorial.py)
- **Tutorial 7**: [Inverse Problem with multiple boundaries](../tutorials/tutorial7/tutorial.py)
- **Tutorial 6**: [Domain operations and geometries](../tutorials/tutorial6/tutorial.py)

## Testing

A comprehensive test suite for multi-domain problems is available in `tests/test_multi_domain_problem.py`.

## Conclusion

PINA's multi-domain support is a powerful feature that enables solving a wide range of complex problems in scientific computing and engineering. The framework handles multi-domain problems seamlessly, allowing you to focus on the physics and mathematics of your problem rather than the implementation details.

For more information, visit the [PINA Documentation](https://mathlab.github.io/PINA/).
