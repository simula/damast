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

# nautical mile in [nmi] = [NM]
nmi = units.def_unit("nmi", 1.852 * units.km)
knots = units.def_unit("knots", nmi / units.hour)

units.add_enabled_units([nmi, knots])

unit_registry = units.get_current_unit_registry().registry
