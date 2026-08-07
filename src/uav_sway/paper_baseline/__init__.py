"""Math-only interfaces for the frozen SEP-NMPC-adapted baseline.

This package deliberately contains no benchmark runner.  It exposes the
planar paper model, the five-link measurement adapter, and the frozen
tracking/passivity/HOCBF contracts used by the future controller.
"""

from .sep_nmpc_model import PlanarParameters, planar_dynamics, planar_mass_matrix

__all__ = ["PlanarParameters", "planar_dynamics", "planar_mass_matrix"]
