"""TreeQSM triangulation subpackage.

Expose the triangulation functions at the package level so that
``from TreeQSMSteps.triangulation import <name>`` binds the callable
function (not just the submodule). Imports are ordered by dependency to
avoid circular-import problems.
"""

from .check_self_intersection import check_self_intersection
from .boundary_curve2 import boundary_curve2
from .boundary_curve import boundary_curve
from .initial_boundary_curve import initial_boundary_curve
from .curve_based_triangulation import curve_based_triangulation

__all__ = [
    "check_self_intersection",
    "boundary_curve2",
    "boundary_curve",
    "initial_boundary_curve",
    "curve_based_triangulation",
]
