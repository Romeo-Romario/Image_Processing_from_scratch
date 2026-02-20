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
    def find_extream_points(self, global_average_threshold: typing.SupportsFloat = 0.7, mean_distance_threshold: typing.SupportsFloat = 0.8) -> list[bool]:
        ...
    def get_clean_text_rows(self) -> list[numpy.typing.NDArray[numpy.float64]]:
        ...
    def get_text_rows(self) -> list[numpy.typing.NDArray[numpy.float64]]:
        ...
    def remove_rows_without_text(self, remove_threshold: typing.SupportsFloat = 8.0, width_threshold: typing.SupportsInt = 40) -> None:
        ...
    def seperate_main_text(self) -> tuple[list[list[float]], list[list[bool]]]:
        ...
    def smooth_row_function(self) -> list[float]:
        ...
