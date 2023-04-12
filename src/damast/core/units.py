"""
This module extends the existing units found in ::class::`astropy.units`.
"""
import astropy.units as _units

__all__ = [
    "knots",
    "units",
    "unit_registry",
    "Unit"
]

units = _units
Unit = _units.Unit
# Additional type
knots = units.def_unit("knots", 1.852 * units.km / units.h)

units.add_enabled_units([knots])

unit_registry = units.get_current_unit_registry().registry
