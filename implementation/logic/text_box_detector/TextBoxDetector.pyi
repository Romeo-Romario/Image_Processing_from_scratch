"""
This module implements logic of TextBoxDetector
"""
from __future__ import annotations
import numpy
import numpy.typing
import typing
__all__: list[str] = ['TextBoxDetector']
class TextBoxDetector:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, deskew_canny_image: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        ...
    def find_extream_points(self) -> list[bool]:
        ...
    def smooth_row_function(self) -> list[float]:
        ...
