"""
This module implements logic of Hough Transform on previously determined edge map
"""
from __future__ import annotations
import numpy
import numpy.typing
import typing
__all__: list[str] = ['HoughTransform']
class HoughTransform:
    def __init__(self, input_edges: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], theta: typing.SupportsFloat = 0.261, rho: typing.SupportsFloat = 9) -> None:
        ...
    def hough_lines(self, threshold: typing.SupportsFloat, min_theta: typing.SupportsFloat, max_theta: typing.SupportsFloat) -> list[list[list[float]]]:
        ...
