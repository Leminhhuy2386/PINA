"""
Test suite for multi-domain problem capabilities in PINA.

This test suite verifies that PINA can handle multi-domain problems correctly,
including:
- Multiple subdomain definitions
- Domain operations (Union, Intersection, Difference)
- Conditions applied to multiple domains
- Interface domains between subdomains
"""

import torch
import pytest
from pina import Condition, LabelTensor
from pina.problem import SpatialProblem
from pina.domain import (
    CartesianDomain,
    EllipsoidDomain,
    Union,
    Difference,
    Intersection,
)
from pina.equation import Equation, FixedValue
from pina.operator import laplacian


def poisson_equation(input_, output_):
    """Simple Poisson equation for testing."""
    laplace_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
    return laplace_u + 1.0


class MultiDomainTestProblem(SpatialProblem):
    """
    Test problem with multiple domains.
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 2], "y": [0, 1]})

    domains = {
        "domain1": CartesianDomain({"x": [0, 1], "y": [0, 1]}),
        "domain2": CartesianDomain({"x": [1, 2], "y": [0, 1]}),
        "interface": CartesianDomain({"x": 1.0, "y": [0, 1]}),
        "boundary_left": CartesianDomain({"x": 0.0, "y": [0, 1]}),
        "boundary_right": CartesianDomain({"x": 2.0, "y": [0, 1]}),
    }

    conditions = {
        "physics_d1": Condition(
            domain="domain1", equation=Equation(poisson_equation)
        ),
        "physics_d2": Condition(
            domain="domain2", equation=Equation(poisson_equation)
        ),
        "bc_left": Condition(domain="boundary_left", equation=FixedValue(0.0)),
        "bc_right": Condition(domain="boundary_right", equation=FixedValue(1.0)),
    }


def test_multi_domain_creation():
    """Test that a multi-domain problem can be created."""
    problem = MultiDomainTestProblem()

    # Check that all domains are registered
    assert len(problem.domains) == 5
    assert "domain1" in problem.domains
    assert "domain2" in problem.domains
    assert "interface" in problem.domains

    # Check that all conditions are registered
    assert len(problem.conditions) == 4
    assert "physics_d1" in problem.conditions
    assert "physics_d2" in problem.conditions


def test_multi_domain_discretization():
    """Test that multiple domains can be discretized independently."""
    problem = MultiDomainTestProblem()

    # Discretize different domains with different numbers of points
    problem.discretise_domain(n=100, mode="random", domains=["domain1"])
    problem.discretise_domain(n=200, mode="random", domains=["domain2"])
    problem.discretise_domain(n=50, mode="random", domains=["interface"])

    # Check that the correct number of points were sampled
    assert problem.discretised_domains["domain1"].shape[0] == 100
    assert problem.discretised_domains["domain2"].shape[0] == 200
    assert problem.discretised_domains["interface"].shape[0] == 50


def test_multi_domain_collected_data():
    """Test that data is correctly collected from multiple domains."""
    problem = MultiDomainTestProblem()

    # Discretize all domains
    problem.discretise_domain(n=50, mode="random", domains="all")

    # Check that collected data includes all conditions
    collected_data = problem.collected_data
    assert "physics_d1" in collected_data
    assert "physics_d2" in collected_data
    assert "bc_left" in collected_data
    assert "bc_right" in collected_data

    # Check that input points are in the correct domains
    d1_pts = collected_data["physics_d1"]["input"]
    assert torch.all(d1_pts.extract("x") >= 0)
    assert torch.all(d1_pts.extract("x") <= 1)

    d2_pts = collected_data["physics_d2"]["input"]
    assert torch.all(d2_pts.extract("x") >= 1)
    assert torch.all(d2_pts.extract("x") <= 2)


def test_domain_operations_multi_domain():
    """Test that domain operations work with multi-domain problems."""

    class DomainOperationProblem(SpatialProblem):
        output_variables = ["u"]

        # Create complex domains using operations
        square1 = CartesianDomain({"x": [0, 1], "y": [0, 1]})
        square2 = CartesianDomain({"x": [0.5, 1.5], "y": [0, 1]})

        spatial_domain = Union([square1, square2])

        domains = {
            "full_domain": Union([square1, square2]),
            "overlap": Intersection([square1, square2]),
            "square1": square1,
            "square2": square2,
        }

        conditions = {
            "physics": Condition(
                domain="full_domain", equation=Equation(poisson_equation)
            )
        }

    problem = DomainOperationProblem()

    # Test that the problem can be created
    assert len(problem.domains) == 4

    # Test discretization of union domain
    problem.discretise_domain(n=100, mode="random", domains=["full_domain"])
    assert problem.discretised_domains["full_domain"].shape[0] == 100


def test_interface_domain_between_subdomains():
    """Test that interface domains can be properly defined and sampled."""
    problem = MultiDomainTestProblem()

    # Discretize the interface
    problem.discretise_domain(n=30, mode="random", domains=["interface"])

    # Check that interface points are at x=1.0
    interface_pts = problem.discretised_domains["interface"]
    assert torch.allclose(
        interface_pts.extract("x"), torch.ones(30), atol=1e-6
    )

    # Check that y coordinates are within bounds
    assert torch.all(interface_pts.extract("y") >= 0)
    assert torch.all(interface_pts.extract("y") <= 1)


def test_multiple_boundary_conditions():
    """Test that multiple boundary conditions can be applied to different domains."""

    class MultiBoundaryProblem(SpatialProblem):
        output_variables = ["u"]
        spatial_domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})

        domains = {
            "interior": CartesianDomain({"x": [0, 1], "y": [0, 1]}),
            "left": CartesianDomain({"x": 0.0, "y": [0, 1]}),
            "right": CartesianDomain({"x": 1.0, "y": [0, 1]}),
            "top": CartesianDomain({"x": [0, 1], "y": 1.0}),
            "bottom": CartesianDomain({"x": [0, 1], "y": 0.0}),
        }

        conditions = {
            "physics": Condition(domain="interior", equation=Equation(poisson_equation)),
            "bc_left": Condition(domain="left", equation=FixedValue(0.0)),
            "bc_right": Condition(domain="right", equation=FixedValue(1.0)),
            "bc_top": Condition(domain="top", equation=FixedValue(0.5)),
            "bc_bottom": Condition(domain="bottom", equation=FixedValue(0.0)),
        }

    problem = MultiBoundaryProblem()

    # Discretize all domains
    problem.discretise_domain(n=50, mode="random", domains="all")

    # Check that all boundary conditions have data
    collected_data = problem.collected_data
    assert len(collected_data) == 5
    assert all(
        bc in collected_data
        for bc in ["bc_left", "bc_right", "bc_top", "bc_bottom"]
    )


def test_different_physics_in_different_domains():
    """Test that different equations can be applied to different domains."""

    def equation1(input_, output_):
        """Equation with source term = 1"""
        laplace_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
        return laplace_u + 1.0

    def equation2(input_, output_):
        """Equation with source term = -1"""
        laplace_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
        return laplace_u - 1.0

    class DifferentPhysicsProblem(SpatialProblem):
        output_variables = ["u"]
        spatial_domain = CartesianDomain({"x": [0, 2], "y": [0, 1]})

        domains = {
            "left_material": CartesianDomain({"x": [0, 1], "y": [0, 1]}),
            "right_material": CartesianDomain({"x": [1, 2], "y": [0, 1]}),
        }

        conditions = {
            "physics_left": Condition(
                domain="left_material", equation=Equation(equation1)
            ),
            "physics_right": Condition(
                domain="right_material", equation=Equation(equation2)
            ),
        }

    problem = DifferentPhysicsProblem()

    # Check that problem is created with different equations
    assert len(problem.conditions) == 2
    assert problem.conditions["physics_left"].equation != problem.conditions[
        "physics_right"
    ].equation


def test_add_points_to_multiple_domains():
    """Test that points can be added to multiple domains."""
    problem = MultiDomainTestProblem()

    # Initial discretization
    problem.discretise_domain(n=10, mode="random", domains=["domain1", "domain2"])

    # Add new points to domain1
    new_pts1 = LabelTensor(torch.tensor([[0.5, 0.5], [0.3, 0.7]]), labels=["x", "y"])
    problem.add_points({"domain1": new_pts1})

    # Add new points to domain2
    new_pts2 = LabelTensor(torch.tensor([[1.5, 0.5], [1.7, 0.3]]), labels=["x", "y"])
    problem.add_points({"domain2": new_pts2})

    # Check that points were added
    assert problem.discretised_domains["domain1"].shape[0] == 12
    assert problem.discretised_domains["domain2"].shape[0] == 12


def test_input_variables_multi_domain():
    """Test that input variables are correctly identified in multi-domain problems."""
    problem = MultiDomainTestProblem()

    # Check input variables
    assert len(problem.input_variables) == 2
    assert "x" in problem.input_variables
    assert "y" in problem.input_variables


def test_output_variables_multi_domain():
    """Test that output variables are correctly identified in multi-domain problems."""
    problem = MultiDomainTestProblem()

    # Check output variables
    assert len(problem.output_variables) == 1
    assert "u" in problem.output_variables


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
