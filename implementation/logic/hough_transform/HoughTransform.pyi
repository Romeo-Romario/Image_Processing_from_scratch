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
    def deskew_image(self, image: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], threshold: typing.SupportsFloat = 2000.0, min_theta: typing.SupportsFloat = -3.141592653589793, max_theta: typing.SupportsFloat = 3.141592653589793) -> numpy.typing.NDArray[numpy.float64]:
        ...
