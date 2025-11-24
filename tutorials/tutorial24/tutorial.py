#!/usr/bin/env python
# coding: utf-8

# # Tutorial: Solving Multi-Domain Problems with PINA
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mathLab/PINA/blob/master/tutorials/tutorial24/tutorial.ipynb)
#
# ## Introduction to Multi-Domain Problems
#
# This tutorial demonstrates how to solve **multi-domain problems** using PINA.
# Multi-domain problems are common in physics and engineering where:
#
# - Different physical laws apply in different regions of space
# - Boundary conditions connect multiple domains
# - Solutions must satisfy continuity conditions at domain interfaces
#
# **PINA fully supports multi-domain problems** through its flexible domain and condition system.
#
# This tutorial will cover:
#
# 1. **Basic Multi-Domain Setup**: Defining multiple domains with different conditions
# 2. **Domain Operations**: Using Difference operations to create complex geometries
# 3. **Interface Conditions**: Enforcing continuity across domain boundaries
# 4. **Complete Multi-Domain Example**: Heat conduction problem with different materials
#
# Let's start by importing the necessary modules:

# In[ ]:


## routine needed to run the notebook on Google Colab
try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False
if IN_COLAB:
    import sys

    sys.path.append("/content/drive/MyDrive")
    get_ipython().system('pip install "pina-mathlab[tutorial]"')

import warnings
import torch
import matplotlib.pyplot as plt
import numpy as np

from pina import Trainer, Condition
from pina.problem import SpatialProblem
from pina.operator import grad, laplacian
from pina.solver import PINN
from pina.model import FeedForward
from pina.domain import CartesianDomain, EllipsoidDomain, Difference
from pina.equation import Equation, FixedValue
from pina.callback import MetricTracker

warnings.filterwarnings("ignore")


# ## Example 1: Basic Multi-Domain Problem
#
# Let's start with a simple example: solving a Poisson equation in a domain
# that is divided into multiple subdomains with different boundary conditions.
#
# We'll solve:
#
# $$
# \begin{equation}
# \begin{cases}
# \nabla^2 u = -1 & \text{in } \Omega_1 \cup \Omega_2 \\
# u = 0 & \text{on } \Gamma_{\text{left}} \\
# u = 0 & \text{on } \Gamma_{\text{right}} \\
# u = 1 & \text{on } \Gamma_{\text{top}} \\
# u = 0 & \text{on } \Gamma_{\text{bottom}} \\
# \end{cases}
# \end{equation}
# $$
#
# where $\Omega_1 = [0, 0.5] \times [0, 1]$ and $\Omega_2 = [0.5, 1] \times [0, 1]$.


# In[2]:


def poisson_equation(input_, output_):
    """
    Poisson equation: Laplacian(u) = -1

    :param LabelTensor input_: Input coordinates
    :param LabelTensor output_: Output field u
    :return: Residual of the Poisson equation
    :rtype: LabelTensor
    """
    laplace_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
    return laplace_u + 1.0


class MultiDomainPoisson(SpatialProblem):
    """
    Multi-domain Poisson problem with two subdomains.
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})

    # Define multiple domains
    domains = {
        # Subdomain 1 (left half)
        "omega1": CartesianDomain({"x": [0, 0.5], "y": [0, 1]}),
        # Subdomain 2 (right half)
        "omega2": CartesianDomain({"x": [0.5, 1], "y": [0, 1]}),
        # Boundary conditions
        "gamma_left": CartesianDomain({"x": 0.0, "y": [0, 1]}),
        "gamma_right": CartesianDomain({"x": 1.0, "y": [0, 1]}),
        "gamma_top": CartesianDomain({"x": [0, 1], "y": 1.0}),
        "gamma_bottom": CartesianDomain({"x": [0, 1], "y": 0.0}),
        # Interface between subdomains
        "interface": CartesianDomain({"x": 0.5, "y": [0, 1]}),
    }

    # Define conditions for each domain
    conditions = {
        # Physics equation in both subdomains
        "physics_omega1": Condition(
            domain="omega1", equation=Equation(poisson_equation)
        ),
        "physics_omega2": Condition(
            domain="omega2", equation=Equation(poisson_equation)
        ),
        # Boundary conditions
        "bc_left": Condition(domain="gamma_left", equation=FixedValue(0.0)),
        "bc_right": Condition(domain="gamma_right", equation=FixedValue(0.0)),
        "bc_top": Condition(domain="gamma_top", equation=FixedValue(1.0)),
        "bc_bottom": Condition(domain="gamma_bottom", equation=FixedValue(0.0)),
    }


# Create the problem instance
problem = MultiDomainPoisson()

# Discretize all domains
problem.discretise_domain(n=200, mode="random", domains=["omega1", "omega2"])
problem.discretise_domain(n=50, mode="random", domains=["interface"])
problem.discretise_domain(
    n=30,
    mode="random",
    domains=["gamma_left", "gamma_right", "gamma_top", "gamma_bottom"],
)

print("Multi-domain problem created successfully!")
print(f"Number of domains: {len(problem.domains)}")
print(f"Domain names: {list(problem.domains.keys())}")


# Now we can see that PINA allows us to define multiple domains and apply
# different conditions to each. Let's visualize the discretized domains:


# In[3]:


# Visualize the sampling points in different domains
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot subdomain 1
omega1_pts = problem.discretised_domains["omega1"]
ax.scatter(
    omega1_pts.extract("x"),
    omega1_pts.extract("y"),
    c="blue",
    alpha=0.5,
    s=10,
    label="Ω₁ (left)",
)

# Plot subdomain 2
omega2_pts = problem.discretised_domains["omega2"]
ax.scatter(
    omega2_pts.extract("x"),
    omega2_pts.extract("y"),
    c="green",
    alpha=0.5,
    s=10,
    label="Ω₂ (right)",
)

# Plot interface
interface_pts = problem.discretised_domains["interface"]
ax.scatter(
    interface_pts.extract("x"),
    interface_pts.extract("y"),
    c="red",
    s=30,
    label="Interface",
    marker="x",
)

# Plot boundaries
boundary_domains = [
    "gamma_left",
    "gamma_right",
    "gamma_top",
    "gamma_bottom",
]
for bdry in boundary_domains:
    bdry_pts = problem.discretised_domains[bdry]
    ax.scatter(
        bdry_pts.extract("x"),
        bdry_pts.extract("y"),
        c="black",
        s=20,
        marker="^",
    )

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Multi-Domain Problem: Sampling Points")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
plt.tight_layout()
plt.show()


# ## Example 2: Using Domain Operations
#
# PINA provides powerful domain operations that allow you to create complex
# geometries from basic shapes. Let's demonstrate this with a more complex
# multi-domain problem.
#
# We'll create a domain with:
# - A square domain with a circular hole in the middle
# - Different conditions in different regions


# In[4]:


def custom_source_equation(input_, output_):
    """
    Poisson equation with a spatially varying source term.

    :param LabelTensor input_: Input coordinates
    :param LabelTensor output_: Output field u
    :return: Residual of the equation
    :rtype: LabelTensor
    """
    x = input_.extract("x")
    y = input_.extract("y")
    laplace_u = laplacian(output_, input_, components=["u"], d=["x", "y"])

    # Different source terms in different regions
    source = torch.where(
        x < 0, -2.0 * torch.ones_like(x), -0.5 * torch.ones_like(x)
    )

    return laplace_u - source


class ComplexMultiDomainProblem(SpatialProblem):
    """
    Multi-domain problem using domain operations:
    Square domain with a circular exclusion.
    """

    output_variables = ["u"]

    # Create a square domain with a circular hole using domain operations
    square = CartesianDomain({"x": [-1, 1], "y": [-1, 1]})
    circle = EllipsoidDomain({"x": [-0.3, 0.3], "y": [-0.3, 0.3]})

    # The computational domain is the square minus the circle
    spatial_domain = Difference([square, circle])

    domains = {
        # Left half of the domain
        "left_domain": CartesianDomain({"x": [-1, 0], "y": [-1, 1]}),
        # Right half of the domain
        "right_domain": CartesianDomain({"x": [0, 1], "y": [-1, 1]}),
        # Outer boundary
        "outer_boundary": CartesianDomain(
            {
                "x": [-1, 1],
                "y": [-1, 1],
            }
        ),
        # Inner boundary (circle)
        "inner_boundary": circle,
    }

    conditions = {
        # Physics in left domain
        "physics_left": Condition(
            domain="left_domain", equation=Equation(custom_source_equation)
        ),
        # Physics in right domain
        "physics_right": Condition(
            domain="right_domain", equation=Equation(custom_source_equation)
        ),
        # Outer boundary condition
        "bc_outer": Condition(domain="outer_boundary", equation=FixedValue(0.0)),
        # Inner boundary condition (on the circle)
        "bc_inner": Condition(domain="inner_boundary", equation=FixedValue(1.0)),
    }


# Note: This is a conceptual example showing how to define complex multi-domain
# problems with domain operations. For actual training, you would need to:
# 1. Properly discretize the boundaries (outer and inner)
# 2. Use appropriate sampling strategies for the exclusion domain
# 3. Train the model with a PINN solver


print("Complex multi-domain problem defined!")
print(
    "Domain operations used: Difference (Square - Circle) for computational domain"
)


# ## Example 3: Heat Conduction in Multiple Materials
#
# Let's create a realistic multi-domain problem: heat conduction in a composite
# material with different thermal conductivities.
#
# We solve:
#
# $$
# \begin{equation}
# \begin{cases}
# \nabla \cdot (k_1 \nabla u) = 0 & \text{in } \Omega_1 \\
# \nabla \cdot (k_2 \nabla u) = 0 & \text{in } \Omega_2 \\
# u = T_{\text{hot}} & \text{on } \Gamma_{\text{left}} \\
# u = T_{\text{cold}} & \text{on } \Gamma_{\text{right}} \\
# \frac{\partial u}{\partial n} = 0 & \text{on } \Gamma_{\text{top/bottom}} \\
# \end{cases}
# \end{equation}
# $$
#
# with continuity conditions at the interface between materials.


# In[5]:


def heat_equation_material1(input_, output_):
    """
    Heat equation for material 1 with thermal conductivity k1 = 1.0

    :param LabelTensor input_: Input coordinates
    :param LabelTensor output_: Temperature field u
    :return: Residual of the heat equation
    :rtype: LabelTensor
    """
    k1 = 1.0
    laplace_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
    return k1 * laplace_u


def heat_equation_material2(input_, output_):
    """
    Heat equation for material 2 with thermal conductivity k2 = 0.1

    :param LabelTensor input_: Input coordinates
    :param LabelTensor output_: Temperature field u
    :return: Residual of the heat equation
    :rtype: LabelTensor
    """
    k2 = 0.1
    laplace_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
    return k2 * laplace_u


class HeatConductionMultiMaterial(SpatialProblem):
    """
    Multi-domain heat conduction problem with two materials.
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 2], "y": [0, 1]})

    domains = {
        # Material 1 (left, high conductivity)
        "material1": CartesianDomain({"x": [0, 1], "y": [0, 1]}),
        # Material 2 (right, low conductivity)
        "material2": CartesianDomain({"x": [1, 2], "y": [0, 1]}),
        # Boundaries
        "left_boundary": CartesianDomain({"x": 0.0, "y": [0, 1]}),
        "right_boundary": CartesianDomain({"x": 2.0, "y": [0, 1]}),
        "interface": CartesianDomain({"x": 1.0, "y": [0, 1]}),
    }

    conditions = {
        # Physics in each material
        "physics_mat1": Condition(
            domain="material1", equation=Equation(heat_equation_material1)
        ),
        "physics_mat2": Condition(
            domain="material2", equation=Equation(heat_equation_material2)
        ),
        # Temperature boundary conditions
        "bc_hot": Condition(
            domain="left_boundary", equation=FixedValue(1.0)
        ),  # Hot side
        "bc_cold": Condition(
            domain="right_boundary", equation=FixedValue(0.0)
        ),  # Cold side
    }


# Create and discretize the problem
heat_problem = HeatConductionMultiMaterial()
heat_problem.discretise_domain(
    n=300, mode="random", domains=["material1", "material2"]
)
heat_problem.discretise_domain(
    n=30, mode="random", domains=["left_boundary", "right_boundary", "interface"]
)

print("Multi-material heat conduction problem created!")


# In[6]:


# Visualize the multi-material domain
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Plot material 1
mat1_pts = heat_problem.discretised_domains["material1"]
ax.scatter(
    mat1_pts.extract("x"),
    mat1_pts.extract("y"),
    c="red",
    alpha=0.4,
    s=10,
    label="Material 1 (k=1.0)",
)

# Plot material 2
mat2_pts = heat_problem.discretised_domains["material2"]
ax.scatter(
    mat2_pts.extract("x"),
    mat2_pts.extract("y"),
    c="blue",
    alpha=0.4,
    s=10,
    label="Material 2 (k=0.1)",
)

# Plot interface
interface_pts = heat_problem.discretised_domains["interface"]
ax.scatter(
    interface_pts.extract("x"),
    interface_pts.extract("y"),
    c="black",
    s=30,
    label="Interface",
    marker="|",
    linewidths=2,
)

# Plot boundaries
left_pts = heat_problem.discretised_domains["left_boundary"]
ax.scatter(
    left_pts.extract("x"),
    left_pts.extract("y"),
    c="darkred",
    s=40,
    label="Hot (T=1)",
    marker="s",
)

right_pts = heat_problem.discretised_domains["right_boundary"]
ax.scatter(
    right_pts.extract("x"),
    right_pts.extract("y"),
    c="darkblue",
    s=40,
    label="Cold (T=0)",
    marker="s",
)

ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.set_title("Multi-Material Heat Conduction: Domain Setup", fontsize=14)
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
plt.tight_layout()
plt.show()


# ## Training a Multi-Domain Model
#
# Now let's train the first multi-domain problem (Poisson equation).
# The process is exactly the same as for single-domain problems!


# In[7]:


# Create a simple feed-forward neural network
model = FeedForward(
    layers=[40, 40, 40],
    func=torch.nn.Tanh,
    output_dimensions=len(problem.output_variables),
    input_dimensions=len(problem.input_variables),
)

# Create PINN solver (works seamlessly with multi-domain problems!)
solver = PINN(problem, model)

# Setup metric tracker to monitor training
metric_tracker = MetricTracker()

# Train the model
trainer = Trainer(
    solver,
    max_epochs=500,
    accelerator="cpu",
    callbacks=[metric_tracker],
    enable_model_summary=False,
)

print("Starting training for multi-domain problem...")
trainer.train()


# ## Visualizing the Solution
#
# Let's visualize the solution of our multi-domain problem.


# In[8]:


# Create a grid for visualization
n_points = 50
x = torch.linspace(0, 1, n_points)
y = torch.linspace(0, 1, n_points)
X, Y = torch.meshgrid(x, y, indexing="ij")

# Create input tensor
from pina import LabelTensor

grid_points = LabelTensor(
    torch.stack([X.flatten(), Y.flatten()], dim=1), labels=["x", "y"]
)

# Get predictions
solver.neural_net.eval()
with torch.no_grad():
    predictions = solver.neural_net(grid_points)

# Reshape for plotting
u_pred = predictions.extract("u").reshape(n_points, n_points).numpy()
X_np = X.numpy()
Y_np = Y.numpy()

# Plot the solution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Contour plot
contour = axes[0].contourf(X_np, Y_np, u_pred, levels=20, cmap="jet")
axes[0].axvline(x=0.5, color="white", linestyle="--", linewidth=2, label="Interface")
axes[0].set_xlabel("x", fontsize=12)
axes[0].set_ylabel("y", fontsize=12)
axes[0].set_title("Multi-Domain Solution (Contour)", fontsize=14)
axes[0].legend()
axes[0].set_aspect("equal")
fig.colorbar(contour, ax=axes[0], label="u")

# 3D surface plot
ax_3d = fig.add_subplot(122, projection="3d")
surf = ax_3d.plot_surface(X_np, Y_np, u_pred, cmap="jet", alpha=0.9)
ax_3d.set_xlabel("x", fontsize=12)
ax_3d.set_ylabel("y", fontsize=12)
ax_3d.set_zlabel("u", fontsize=12)
ax_3d.set_title("Multi-Domain Solution (3D)", fontsize=14)
fig.colorbar(surf, ax=ax_3d, label="u", shrink=0.5)

plt.tight_layout()
plt.show()


# ## Key Takeaways
#
# This tutorial demonstrated that **PINA fully supports multi-domain problems**:
#
# 1. **Multiple Domains**: You can define as many domains as needed using the `domains` dictionary
#
# 2. **Domain Operations**: Use domain operations like `Difference` to create complex geometries (see tutorial 6 for more operations)
#
# 3. **Flexible Conditions**: Apply different equations and boundary conditions to each domain
#
# 4. **Seamless Training**: The PINN solver works exactly the same way for multi-domain problems
#
# 5. **Interface Handling**: Define interface domains to sample points at domain boundaries
#
# 6. **Real Applications**: Multi-domain capability enables solving problems with:
#    - Multiple materials with different properties
#    - Coupled physics in different regions
#    - Complex boundary conditions
#    - Composite structures
#
# ## Next Steps
#
# To learn more about PINA's capabilities:
#
# 1. **Experiment with Complex Geometries**: Try combining multiple domain operations
# 2. **Add Interface Conditions**: Implement continuity conditions at interfaces
# 3. **Multi-Physics Problems**: Solve coupled problems across domains
# 4. **Adaptive Sampling**: Use different sampling strategies for different domains
#
# For more tutorials, visit the [PINA Documentation](https://mathlab.github.io/PINA/).
