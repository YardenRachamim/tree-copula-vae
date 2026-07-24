import torch
from torch.distributions import constraints
from typing import Union


class _UnitCube(constraints.Constraint):
    def __init__(self):
        self.lower_bound = 0.
        self.upper_bound = 1.
        super().__init__()

    def check(self, value):
        is_lower_bound = (value >= self.lower_bound).all()
        is_upper_bound = (value <= self.upper_bound).all()

        return is_lower_bound and is_upper_bound

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += '(lower_bound={}, upper_bound={})' \
            .format(self.lower_bound, self.upper_bound)

        return fmt_string


class _GreaterThanEqAndDifferentFrom(constraints.Constraint):
    def __init__(self, lower_bound, different_from):
        self.lower_bound = lower_bound
        self.greater_then_eq = constraints.greater_than_eq(lower_bound)
        self.different_from = different_from
        super().__init__()

    def check(self, value):
        return (self.different_from != value) and self.greater_then_eq.check(value)

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += '(lower_bound={}, different_from={})' \
            .format(self.lower_bound, self.different_from)

        return fmt_string


class _CorrelationMatrix(constraints.Constraint):
    def __init__(self):
        self.lower_bound = -1
        self.upper_bound = 1

        self.interval = constraints.interval(self.lower_bound, self.upper_bound)

    def check(self, value):
        # Check if the matrix is positive definite
        is_pd = constraints.positive_definite.check(value)
        is_valid_diagonal = torch.diagonal(value, dim1=-2, dim2=-1, offset=0) == 1.

        # Check if the elements are in the specified range
        is_in_range = self.interval.check(value)

        return is_pd & is_in_range & is_valid_diagonal

    def __repr__(self):
        fmt_string = self.__class__.__name__[1:]
        fmt_string += '(lower_bound={}, upper_bound={})' \
            .format(self.lower_bound, self.upper_bound)

        return fmt_string


greater_then_eq_and_different_from = _GreaterThanEqAndDifferentFrom
unit_cube = _UnitCube()
is_correlation_matrix = _CorrelationMatrix()
