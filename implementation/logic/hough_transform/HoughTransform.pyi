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
    def deskew(self, image: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], rotation_matrix: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
        ...
    def get_deskew_angle(self, threshold: typing.SupportsFloat, min_theta: typing.SupportsFloat, max_theta: typing.SupportsFloat) -> float:
        ...
    def get_rotation_matrix(self, center: tuple[typing.SupportsInt, typing.SupportsInt], angle: typing.SupportsFloat, scale: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]:
        ...
    def hough_lines(self, threshold: typing.SupportsFloat, min_theta: typing.SupportsFloat, max_theta: typing.SupportsFloat) -> list[list[list[float]]]:
        ...
