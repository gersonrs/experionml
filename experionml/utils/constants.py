from __future__ import annotations

__all__ = [
    "CAT_TYPES",
    "COLOR_SCHEME",
    "DEFAULT_MISSING",
    "DF_ATTRS",
    "PALETTE",
    "__version__",
]

# Current library version
__version__ = "1.5.0"  # x-release-please-version

# Column types considered categorical
CAT_TYPES = ["object", "category", "string", "boolean"]

# Default string values considered missing
DEFAULT_MISSING = ["", "?", "NA", "nan", "NaN", "NaT", "none", "None", "inf", "-inf"]

# Attributes shared between experionml and a dataframe
DF_ATTRS = (
    "size",
    "head",
    "tail",
    "loc",
    "iloc",
    "describe",
    "iterrows",
    "dtypes",
    "at",
    "iat",
    "memory_usage",
    "empty",
    "ndim",
)

# Highlighted color scheme for styler objects
COLOR_SCHEME = "background-color: lightblue"

# Default color palette (discrete color, continuous scale)
PALETTE = {
    "rgb(0, 63, 136)": "Blues",
    "rgb(31, 119, 180)": "Blues",
    "rgb(72, 160, 220)": "Blues",
    "rgb(100, 193, 232)": "GnBu",
    "rgb(44, 95, 160)": "Blues",
    "rgb(90, 50, 140)": "Purples",
    "rgb(140, 86, 75)": "Oranges",
    "rgb(64, 64, 128)": "Purples",
    "rgb(102, 102, 102)": "Greys",
}
