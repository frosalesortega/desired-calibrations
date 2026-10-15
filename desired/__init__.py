"""
DESIRED — Strong-Line Metallicity Calibrations
===============================================

DEep Spectra of Ionised REgions Database (DESIRED)
Rosales-Ortega et al. (2026a)

Quick start
-----------
>>> from desired import calibrate, list_calibrations
>>> # See all available calibrations
>>> list_calibrations()
>>> # Apply a monotonic calibration
>>> OH_t2eq0, OH_t2eq0_e, OH_t2ge0, OH_t2ge0_e = calibrate('N2', -1.2, index_e=0.05)
>>> # Apply a double-valued calibration (upper branch)
>>> OH_t2eq0, OH_t2eq0_e, OH_t2ge0, OH_t2ge0_e = calibrate('R3', 0.6, branch='upper')
"""

__version__ = "2.0.0"
__author__  = "F. F. Rosales-Ortega"
__reference__ = "Rosales-Ortega et al. (2026a)"

from .calibrate import calibrate, list_calibrations, get_info

__all__ = ["calibrate", "list_calibrations", "get_info"]
