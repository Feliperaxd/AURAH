# -----------------------------------------------------------------------------
# AURAH - Analysis and Understanding of Human Attributes
# Copyright (c) 2023 Felipe Amaral dos Santos
# Licensed under the MIT License (see LICENSE file)
# -----------------------------------------------------------------------------

__all__ = [
    "AnyKey",
    "AnyValue",
    "CoordinatesFloat",
    "CoordinatesInt",
    "DataPack",
    "HeightFloat",
    "HeightInt",
    "SizeFloat",
    "SizeInt",
    "WidthFloat",
    "WidthInt",
    "XCoordinateFloat",
    "XCoordinateInt",
    "YCoordinateFloat",
    "YCoordinateInt",
    "ZoneFloat",
    "ZoneInt"    
]

from typing import Dict, List, Tuple, TypeVar


# Base types for coordinates
XCoordinateFloat = float
XCoordinateInt = int
YCoordinateFloat = float
YCoordinateInt = int

# Base types for dimensions
WidthFloat = float
WidthInt = int
HeightFloat = float
HeightInt = int

# Generic type variables
K = TypeVar('K')
V = TypeVar('V')

# Advanced types for coordinates
CoordinatesFloat = Tuple[XCoordinateFloat, YCoordinateFloat]
CoordinatesInt = Tuple[XCoordinateInt, YCoordinateInt]
ZoneFloat = Tuple[XCoordinateFloat, YCoordinateFloat, WidthFloat, HeightFloat]
ZoneInt = Tuple[XCoordinateInt, YCoordinateInt, WidthInt, HeightInt]

# Advanced types for dimensions
SizeFloat = Tuple[WidthFloat, HeightFloat]
SizeInt = Tuple[WidthInt, HeightInt]

# Advanced types for storage
DataPack = Dict[K, List[V]]
