"""
Python adaptation and extension of TREEQSM.

Version: 0.0.4
Date: 4 March 2025
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""
import numpy as np
try:
    from ..Utils import Utils
except ImportError:
    import Utils.Utils as Utils
import numba

# class LSF:


def rotate_to_z_axis(Vec):
    """
    Forms the rotation matrix to rotate the vector Vec to a point along the positive z-axis.

    Args:
        Vec : A 3-element vector.

    Returns:
        (numpy.ndarray, numpy.ndarray, float): Rotation matrix, axis of rotation, and angle.
    """
    Vec = np.array(Vec, dtype=float).flatten()
    D = np.cross(Vec, [0, 0, 1])
    if np.linalg.norm(D) > 0:
        # Use acos on the z-component (assuming Vec is normalized or that we want the angle with z-axis)
        a = np.arccos(Vec[2])
        R = Utils.rotation_matrix(D, a)
    else:
        R = np.eye(3)
        a = 0.0
        D = np.array([1.0, 0.0, 0.0])
    return R, D, a

@numba.jit()
def form_rotation_matrices(theta):
    """
    Forms rotation matrices R = R2*R1 and computes the derivatives dR1 and dR2.

    Args:
        theta: An array-like with two elements [t1, t2] representing the plane rotation angles.

    Returns:
        (numpy.ndarray, numpy.ndarray, numpy.ndarray): Rotation matrix R, derivative of R1 with respect to t1, and derivative of R2 with respect to t2.
    """
    theta = theta.flatten()
    if theta.size < 2:
        raise ValueError("theta must contain at least two elements")

    c = np.cos(theta)
    s = np.sin(theta)

    # Plane rotation R1: rotation about the x-axis by angle t1
    R1 = np.array((
        (1,     0,     0),
        (0,   c[0], -s[0]),
        (0,   s[0],  c[0])
        ))

    # Plane rotation R2: rotation about the y-axis by angle t2
    R2 = np.array((
        (c[1],  0,  s[1]),
        (0,     1,  0   ),
        (-s[1], 0,  c[1])
    ))

    R = R2 @ R1  # matrix multiplication

    # Derivative of R1 with respect to t1:
    # MATLAB: dR1 = [0 0 0; 0 -R1(3,2) -R1(2,2); 0 R1(2,2) -R1(3,2)];
    # Note: MATLAB indexing: R1(3,2) -> Python: R1[2,1], R1(2,2) -> Python: R1[1,1]
    dR1 = np.array((
        (0,       0,      0),
        (0,   -R1[2,1], -R1[1,1]),
        (0,    R1[1,1], -R1[2,1])
    ))

    # Derivative of R2 with respect to t2:
    # MATLAB: dR2 = [-R2(1,3) 0 R2(1,1); 0 0 0; -R2(1,1) 0 -R2(1,3)];
    # MATLAB indexing: R2(1,3) -> Python: R2[0,2], R2(1,1) -> Python: R2[0,0]
    dR2 = np.array((
        (-R2[0,2],  0,  R2[0,0]),
        (0,         0,  0),
        (-R2[0,0],  0, -R2[0,2])
        ))

    return R, dR1, dR2


def func_grad_axis(P, par, weight=None):
    """
    Function and gradient calculation for least-squares cylinder fit.

    Cylinder parameters (par):
        [x0, y0, alpha, beta, r]
        where (x0, y0) is the cylinder axis intercept with the xy-plane (with z0 = 0),
        alpha and beta are rotation angles (in radians) about x and y axes, and
        r is the cylinder radius.

    For a point P (n x 3), the transformed point is:
        Pt = (P - [x0, y0, 0]) @ R.T
    where R is the rotation matrix computed as R = R2 * R1 (from angles [alpha, beta]).

    The signed distance for each point is defined as:
        dist = sqrt(xt^2 + yt^2) - r
    where (xt, yt) are the first two coordinates of Pt.

    Optionally, if a weight vector is provided, the distances and the Jacobian are weighted.
    Args:
        P (numpy.ndarray): Point cloud, shape (n_points, 3).
        par (list or numpy.ndarray): Cylinder parameters [x0, y0, alpha, beta, r].
        weight (numpy.ndarray, optional): Weights for each point, shape (n_points,).

    Returns:
        (numpy.ndarray, numpy.ndarray):  Distances and Jacobian matrix.
    """
    par = np.array(par, dtype=float).flatten()
    if par.size != 5:
        raise ValueError("Parameter vector 'par' must have 5 elements: [x0, y0, alpha, beta, r].")

    x0, y0, alpha, beta, r = par
    # Compute rotation matrices and their derivatives using our own implementation.
    R, DR1, DR2 = form_rotation_matrices(np.array([alpha, beta]))

    # Compute the transformed points:
    # Subtract [x0, y0, 0] from each point
    P_shifted = P - np.array([x0, y0, 0])
    # MATLAB uses (P - [x0 y0 0]) * R', so in Python we use dot with R.T
    Pt = P_shifted @ R.T

    xt = Pt[:, 0]
    yt = Pt[:, 1]
    rt_vals = np.sqrt(xt**2 + yt**2)
    # Signed distance: distance from the cylinder surface.
    dist = rt_vals - r
    if weight is not None:
        weight = np.array(weight, dtype=float).flatten()
        dist = weight * dist

    # Compute Jacobian only if needed
    J = None
    # Compute Jacobian with respect to alpha and beta (2 parameters)
    # N: unit vector in the direction of [xt, yt]
    # Avoid division by zero by using np.where
    with np.errstate(divide='ignore', invalid='ignore'):
        N = np.column_stack((np.divide(xt, rt_vals, where=rt_vals!=0),
                                np.divide(yt, rt_vals, where=rt_vals!=0)))
        # For any zero values, set to zero (should not occur in practice if points are not on the axis)
        N[np.isnan(N)] = 0.0

    m = P.shape[0]
    J = np.zeros((m, 2))

    # Compute A3 = (P - [x0, y0, 0]) @ DR1.T and use its first two columns.
    A3 = P_shifted @ DR1.T
    J[:, 0] = np.sum(N * A3[:, :2], axis=1)

    # Compute A4 = (P - [x0, y0, 0]) @ DR2.T and use its first two columns.
    A4 = P_shifted @ DR2.T
    J[:, 1] = np.sum(N * A4[:, :2], axis=1)

    if weight is not None:
        # Weight the Jacobian columns.
        J = np.column_stack((weight * J[:, 0], weight * J[:, 1]))

    return dist, J


def func_grad_circle(P, par, weight=None):
    """
    Function and gradient calculation for least-squares circle fit.

    Args:
        P      : (n x m) array representing the point cloud (only first two columns are used)
        par    : Circle parameters [x0, y0, r]
        weight : Optional (n,) array of weights to be applied to the distances

    Returns:
        (numpy.ndarray, numpy.ndarray): distances and Jacobian matrix
        
    """
    par = np.array(par, dtype=np.float64).flatten()
    if par.size != 3:
        raise ValueError("Parameter vector 'par' must have 3 elements: [x0, y0, r].")

    # Compute differences for x and y coordinates.
    Vx = P[:, 0] - par[0]
    Vy = P[:, 1] - par[1]
    rt = np.sqrt(Vx**2 + Vy**2)

    # Compute distance: rt - r, weighted if needed.
    dist = rt - par[2]
    if weight is not None:
        weight = np.array(weight, dtype=float).flatten()
        dist = weight * dist

    # Form the Jacobian matrix if requested.
    # Avoid division by zero: if rt==0, set derivative to 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        J0 = -np.divide(Vx, rt, where=(rt != 0))
        J1 = -np.divide(Vy, rt, where=(rt != 0))
    # For any zero denominators, fill in zeros.
    J0[~np.isfinite(J0)] = 0.0
    J1[~np.isfinite(J1)] = 0.0

    m = P.shape[0]
    J = np.column_stack((J0, J1, -np.ones(m)))
    if weight is not None:
        J = np.column_stack((weight * J[:, 0], weight * J[:, 1], weight * J[:, 2]))

    return dist, J


def func_grad_circle_centre(P, par, weight=None):
    """
    Function and gradient calculation for least-squares circle fit with respect to circle center only.

    Args:
        P      : (n x m) array representing the point cloud (only first two columns are used)
        par    : Circle parameters [x0, y0, r]
        weight : Optional (n,) array of weights for the points. Distances and derivatives are weighted if provided.

    Returns:
        (numpy.ndarray, numpy.ndarray): distances and Jacobian matrix
    """
    par = np.array(par, dtype=float).flatten()
    if par.size != 3:
        raise ValueError("Parameter vector 'par' must have 3 elements: [x0, y0, r].")
    x0, y0, r = par
    Vx = P[:, 0] - x0
    Vy = P[:, 1] - y0
    rt_vals = np.sqrt(Vx**2 + Vy**2)
    dist = rt_vals - r
    if weight is not None:
        weight = np.array(weight, dtype=float).flatten()
        dist = weight * dist

    with np.errstate(divide='ignore', invalid='ignore'):
        J0 = -np.divide(Vx, rt_vals, where=(rt_vals != 0))
        J1 = -np.divide(Vy, rt_vals, where=(rt_vals != 0))
    J0[~np.isfinite(J0)] = 0.0
    J1[~np.isfinite(J1)] = 0.0

    J = np.column_stack((J0, J1))
    if weight is not None:
        J = np.column_stack((weight * J[:, 0], weight * J[:, 1]))
    return dist, J


@numba.jit()
def func_grad_cylinder(par, P, weight=None):
    """
    Function and gradient calculation for least-squares cylinder fit.

    Args:
        par    : Cylinder parameters [x0, y0, alpha, beta, r]
                    where (x0, y0) is the cylinder axis intercept with the xy-plane (z0 = 0),
                    alpha and beta are rotation angles (in radians) about the x and y axes,
                    and r is the cylinder radius.
        P      : (n x 3) point cloud.
        weight : Optional (n,) weights for the points; if provided, the distances and
                    the Jacobian are weighted.

    Returns:
        (np.ndarray, np.ndarray): Distances and Jacobian matrix.
    """
    if weight is None:
        weight = np.ones(len(P))
    par = par.flatten()
    if par.size != 5:
        raise ValueError("Parameter vector 'par' must have 5 elements: [x0, y0, alpha, beta, r].")
    x0, y0, alpha, beta, r = par

    # Get rotation matrices and their derivatives.
    R, DR1, DR2 = form_rotation_matrices(np.array([alpha, beta]))

    # Transform the point cloud.
    P_shifted = P - np.array([x0, y0, 0])
    # MATLAB: Pt = (P - [x0 y0 0]) * R'  --> in Python: Pt = P_shifted @ R.T
    Pt = P_shifted @ R.T
    xt = Pt[:, 0]
    yt = Pt[:, 1]
    rt_vals = np.sqrt(xt**2 + yt**2)

    # Compute distances.
    dist = rt_vals - r
    if weight is not None:
        weight = weight.flatten()
        dist = weight * dist

    # Build the Jacobian matrix.
    # Compute unit vector N = [xt, yt] / rt.
    # np.seterr(divide='ignore')
    # np.seterr(invalid='ignore')
    # with np.errstate(divide='ignore', invalid='ignore'):
    I = rt_vals==0
    rt_vals[I]=1
    out =np.zeros((len(xt),2), dtype=np.float64)
    # out1 = np.zeros(len(xt), dtype=np.float64)
    # out2 = np.zeros(len(xt), dtype=np.float64)
    N1=np.divide(xt, rt_vals)#,rt_vals != 0)
    N2=np.divide(yt, rt_vals)#,rt_vals != 0)
    # N1[N1==np.nan] = 0.0  # Handle any division by zero
    # N2[N2==np.nan] = 0.0  # Handle any division by zero
    N1[I]=0
    N2[I]=0
    out[:,0]=N1
    out[:,1]=N2
    N = out
    

    m = P.shape[0]
    J = np.zeros((m, 5))

    # Derivative with respect to x0.
    # A1 = (R * [-1,0,0]')'  --> compute R @ [-1,0,0] as a column, then flatten.
    A1 = (R @ np.array([-1, 0, 0],dtype =np.float64).reshape(3, 1)).flatten()
    # Only use first two components.
    # MATLAB: A = eye(2); A(1,1)=A1(1); A(2,2)=A1(2); then J(:,1)= sum( N * A, 2 )
    J[:, 0] = N[:, 0] * A1[0] + N[:, 1] * A1[1]

    # Derivative with respect to y0.
    A2 = (R @ np.array([0, -1, 0],dtype=np.float64).reshape(3, 1)).flatten()
    J[:, 1] = N[:, 0] * A2[0] + N[:, 1] * A2[1]

    # Derivative with respect to alpha.
    A3 = P_shifted @ DR1.T
    J[:, 2] = np.sum(N * A3[:, :2], axis=1)

    # Derivative with respect to beta.
    A4 = P_shifted @ DR2.T
    J[:, 3] = np.sum(N * A4[:, :2], axis=1)

    # Derivative with respect to r.
    J[:, 4] = -1.0

    if weight is not None:
        J = np.column_stack((weight * J[:, 0],
                                weight * J[:, 1],
                                weight * J[:, 2],
                                weight * J[:, 3],
                                weight * J[:, 4]))
    return dist, J


def nlssolver(par, P, weight=None):
    """
    Nonlinear least squares solver for cylinders using Gauss–Newton iterations.

    Args:
        par (np.Array): Initial estimates of the cylinder parameters [x0, y0, alpha, beta, r].
        P (np.Array): (n x 3) point cloud.
        weight (np.Array): Optional (n,) array of weights for the points.

    Output:

        (numpy.ndarray, numpy.ndarray, bool, bool): parameters, distances, convergence status, and reliability status (conditions were acceptable).
    """
    maxiter = 50
    iter_count = 0
    conv = False
    rel = True
    par = np.array(par,dtype=np.float64)
    NoWeights = (weight is None)
    # Gauss–Newton iterations
    while iter_count < maxiter and not conv and rel:
        # Calculate distances and Jacobian using func_grad_cylinder.
        if NoWeights:
            d0, J = func_grad_cylinder(par, P)
        else:
            d0, J = func_grad_cylinder(par, P, weight)

        SS0 = np.linalg.norm(d0)  # norm of distances

        # Solve the linear system: p = - (J^T J)^{-1} J^T d0
        A = J.T @ J
        b = J.T @ d0
        # Use a solver; if A is singular, an exception will be raised.
        try:
            p = -np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            rel = False
            break
        par = par + p  # update parameters

        # Check reliability: using reciprocal condition number of A.
        rcond_val = 1.0 / np.linalg.cond(A) if A.size and np.linalg.cond(A) != 0 else 0.0
        if rcond_val < 10000 * np.finfo(float).eps:
            rel = False

        # Check convergence by comparing norm of distances.
        if NoWeights:
            d_new, _ = func_grad_cylinder(par, P)
        else:
            d_new, _ = func_grad_cylinder(par, P, weight)
        SS1 = np.linalg.norm(d_new)
        if abs(SS0 - SS1) < 1e-4:
            conv = True

        iter_count += 1

    # After iterations, return final distances (only the first output from func_grad_cylinder)
    if NoWeights:
        d, _ = func_grad_cylinder(par, P)
    else:
        d, _ = func_grad_cylinder(par, P, weight)
    return par, d, conv, rel


def least_squares_axis(P, Axis, Point0, Rad0, weight=None):
    """
    Least-squares cylinder axis fitting using Gauss–Newton iterations.
    The fitting is performed on the rotation angles (alpha and beta) only,
    while the cylinder’s axis point (Point0) and radius (Rad0) remain fixed.

    Args:
        P (np.Array): (n x 3) 3D point cloud.
        Axis (np.Array): (3,) initial axis estimate.
        Point0 (np.Array): (3,) initial axis point.
        Rad0 (float): initial cylinder radius.
        weight (np.array): Optional (n,) weights for each point.

    Returns:
        dictionary: Dictionary with fields:
                    - 'axis'   : Optimized cylinder axis (unit vector, row vector).
                    - 'radius' : Cylinder radius (input Rad0).
                    - 'start'  : Cylinder axis point (input Point0).
                    - 'mad'    : Mean absolute distance of the points to the cylinder surface.
                    - 'SurfCov': Surface coverage (fraction of the cylinder surface covered by points).
                    - 'conv'   : True if the Gauss–Newton fitting converged.
                    - 'rel'    : True if the system matrix was well conditioned.
    """
    res = 0.03  # Resolution level for computing surface coverage
    # Initially, the free parameters (rotation angles) are zero.
    par_free = np.zeros(2)
    maxiter = 50
    iter_count = 0
    conv = False
    rel = True

    if weight is None:
        weight = np.ones(P.shape[0])

    # Transform the point cloud so that the axis is aligned with the positive z-axis.
    # Rot0 is the rotation matrix to align Axis with z.
    Rot0, _, _ = rotate_to_z_axis(Axis)
    Pt = (P - Point0) @ Rot0.T

    # Initial full parameter vector: first two parameters (x0,y0) fixed to zero,
    # next two are the free rotation angles (alpha, beta), and the last is the radius.
    Par = np.concatenate((np.zeros(2), par_free, [Rad0]))

    # Gauss–Newton iterations.
    while iter_count < maxiter and (not conv) and rel:
        # Calculate distances and Jacobian using the transformed points.
        # Note: func_grad_axis works on the rotated point cloud Pt.
        d0, J = func_grad_axis(Pt, Par)
        SS0 = np.linalg.norm(d0)

        A = J.T @ J
        b = J.T @ d0
        try:
            p_update = -np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            rel = False
            break

        # Update the free parameters (only alpha and beta are free).
        par_free = par_free + p_update
        Par = np.concatenate((np.zeros(2), par_free, [Rad0]))

        # Recalculate distances with updated parameters.
        d_new, _ = func_grad_axis(Pt, Par)
        SS1 = np.linalg.norm(d_new)
        if SS1 > SS0:
            # If error increased, reduce the step size.
            par_free = par_free - 0.95 * p_update
            Par = np.concatenate((np.zeros(2), par_free, [Rad0]))
            d_new, _ = func_grad_axis(Pt, Par)
            SS1 = np.linalg.norm(d_new)

        # Check reliability: if the reciprocal condition number of A is too low.
        rcond_val = 1.0 / np.linalg.cond(A) if np.linalg.cond(A) != 0 else 0.0
        if rcond_val < 10000 * np.finfo(float).eps:
            rel = False

        # Check convergence.
        if abs(SS0 - SS1) < 1e-5:
            conv = True

        iter_count += 1

    # After iteration, compute the optimized axis.
    # The free parameters are the optimized [alpha, beta] in par_free.
    # Compute the rotation matrix R corresponding to [alpha, beta].
    R_opt, _, _ = form_rotation_matrices(par_free)
    # Transform back: the final axis is given by Rot0' * R_opt' * [0,0,1]'
    Axis_opt = Rot0.T @ R_opt.T @ np.array([0, 0, 1])

    # Compute distances from the original points to the fitted axis.
    # Use the utility distances_to_line: P, axis, and Point0.
    d_axis, _, h, _ = Utils.distances_to_line(P, Axis_opt, Point0)
    d_axis = d_axis - Rad0  # subtract cylinder radius
    Len = np.max(h) - np.min(h)

    # Compute mean absolute deviation (mad) using points with maximum weight.
    if np.all(weight == weight[0]):
        mad = np.mean(np.abs(d_axis))
    else:
        I = (weight == np.max(weight))
        mad = np.mean(np.abs(d_axis[I]))

    # Compute surface coverage if parameters are valid and fit converged.
    if (not np.any(np.isnan(par_free))) and rel and conv:
        nl = int(np.ceil(Len / res))
        nl = max(nl, 3)
        ns = int(np.ceil(2 * np.pi * Rad0 / res))
        ns = max(ns, 8)
        ns = min(ns, 36)
        # Utils.surface_coverage returns a tuple; we take the first element.
        SurfCov, _, _, _ = Utils.surface_coverage(P, Axis_opt, Point0, nl, ns, 0.8 * Rad0)
        SurfCov = float(SurfCov)
    else:
        SurfCov = 0.0

    # Prepare the output cylinder structure (as a dictionary).
    cyl = {
        'radius': Rad0,
        'start': Point0,
        'axis': Axis_opt.flatten(),  # as a row vector
        'mad': mad,
        'SurfCov': SurfCov,
        'conv': conv,
        'rel': rel
    }
    return cyl


def least_squares_circle(P, Point0, Rad0, weight=None):
    """
    Least-squares circle fitting using Gauss–Newton iterations.

    Input:
        P (np.Array): (n x 2) point cloud.
        Point0 (np.Array): (2,) initial estimate of the centre.
        Rad0 (float): initial estimate of the circle radius.
        weight (np.Array): Optional (n,) weights for each point.

    Returns:
        dictionary : A dictionary with the following fields:
                    - 'radius' : Fitted circle radius.
                    - 'point'  : Fitted centre (as a row vector).
                    - 'mad'    : Mean absolute distance of the points to the circle.
                    - 'ArcCov' : Fraction of the circle arc covered by points.
                    - 'conv'   : True if the algorithm converged.
                    - 'rel'    : True if the system matrix was well conditioned.
    """
    # Initial parameter vector: [x0, y0, r]
    par = np.concatenate((np.array(Point0, dtype=float).flatten(), [float(Rad0)]))
    maxiter = 200
    iter_count = 0
    conv = False
    rel = True
    if weight is None:
        weight = np.ones(P.shape[0])
    else:
        weight = np.array(weight, dtype=float).flatten()

    # Gauss–Newton iterations.
    while iter_count < maxiter and (not conv) and rel:
        # Compute distances and Jacobian using func_grad_circle.
        dist, J = func_grad_circle(P, par, weight)
        SS0 = np.linalg.norm(dist)
        A = J.T @ J
        b = J.T @ dist
        try:
            p = -np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            rel = False
            break
        par = par + p
        dist, _ = func_grad_circle(P, par, weight)
        SS1 = np.linalg.norm(dist)
        if SS1 > SS0:
            par = par - 0.95 * p
            dist, _ = func_grad_circle(P, par, weight)
            SS1 = np.linalg.norm(dist)
        if np.linalg.cond(A) != 0 and (1.0/np.linalg.cond(A)) < 10000 * np.finfo(float).eps:
            rel = False
        if abs(SS0 - SS1) < 1e-5:
            conv = True
        iter_count += 1

    # Final parameters.
    Rad = par[2]
    Point = par[0:2]
    U = P[:, 0] - Point[0]
    V = P[:, 1] - Point[1]
    dist_final = np.sqrt(U**2 + V**2) - Rad
    # Compute mean absolute deviation (mad)
    if weight is None or np.allclose(weight, weight[0]):
        mad = np.mean(np.abs(dist_final))
    else:
        I = (weight == np.max(weight))
        mad = np.mean(np.abs(dist_final[I]))

    # Compute arc coverage.
    if not np.any(np.isnan(par)):
        if weight is None or np.allclose(weight, weight[0]):
            I_arc = dist_final > -0.2 * Rad
        else:
            I_arc = (dist_final > -0.2 * Rad) & (weight == np.max(weight))
        U_filt = U[I_arc]
        V_filt = V[I_arc]
        # Compute angles in [0,2pi)
        ang = np.arctan2(V_filt, U_filt) + np.pi
        # Map angles to 1,...,100
        ang_scaled = np.ceil(ang / (2 * np.pi) * 100)
        ang_scaled[ang_scaled <= 0] = 1
        # Create an arc indicator vector (length 100)
        Arc = np.zeros(100, dtype=bool)
        # Subtract 1 for zero-based indexing
        Arc[(ang_scaled - 1).astype(int)] = True
        ArcCov = np.count_nonzero(Arc) / 100.0
    else:
        ArcCov = 0.0

    cir = {
        'radius': Rad,
        'point': Point.flatten(),
        'mad': mad,
        'ArcCov': ArcCov,
        'conv': conv,
        'rel': rel
    }
    return cir


def least_squares_circle_centre(P, Point0, Rad0):
    """
    Least-squares circle fitting such that the radius is fixed (only the centre is optimized)
    using Gauss–Newton iterations.

    Args:
        P (np.Array): (n x 2) array representing the 2D point cloud.
        Point0 (np.Array): (2,) initial estimate of the circle centre.
        Rad0 (float): Given circle radius.

    Output:
        dictionary   : Dictionary with the following fields:
                    - 'radius' : The fixed circle radius (Rad0).
                    - 'point'  : Fitted centre point (as a row vector).
                    - 'mad'    : Mean absolute deviation of the points from the circle.
                    - 'ArcCov' : Arc coverage, the fraction of 100 angle bins covered.
                    - 'conv'   : Boolean flag, True if the algorithm converged.
                    - 'rel'    : Boolean flag, True if the system matrix was well conditioned.
    """
    # Initialize parameter vector: [x0, y0, r]
    par = np.concatenate((np.array(Point0, dtype=float).flatten(), [float(Rad0)]))
    maxiter = 200
    iter_count = 0
    conv = False
    rel = True

    # Gauss–Newton iterations
    while iter_count < maxiter and (not conv) and rel:
        # Compute distances and Jacobian (for centre only, using func_grad_circle_centre)
        dist, J = func_grad_circle_centre(P, par)
        SS0 = np.linalg.norm(dist)
        A = J.T @ J
        b = J.T @ dist
        try:
            p = -np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            rel = False
            break
        # Update only the centre (first two elements); radius remains fixed.
        par[0:2] = par[0:2] + p
        dist, _ = func_grad_circle_centre(P, par)
        SS1 = np.linalg.norm(dist)
        if SS1 > SS0:
            par[0:2] = par[0:2] - 0.95 * p
            dist, _ = func_grad_circle_centre(P, par)
            SS1 = np.linalg.norm(dist)
        if np.linalg.cond(A) != 0 and (1.0 / np.linalg.cond(A)) < 10000 * np.finfo(float).eps:
            rel = False
        if abs(SS0 - SS1) < 1e-4:
            conv = True
        iter_count += 1

    # Compute output: centre, mad, and arc coverage.
    Point = par[0:2]
    U = P[:, 0] - Point[0]
    V = P[:, 1] - Point[1]
    # Compute angles in [0, 2*pi)
    ang = np.arctan2(V, U) + np.pi
    # Map angles to 1,...,100 bins
    ang_bins = np.ceil(ang / (2 * np.pi) * 100)
    ang_bins[ang_bins <= 0] = 1
    Arc = np.zeros(100, dtype=bool)
    # Adjust for zero-based indexing
    Arc[(ang_bins - 1).astype(int)] = True
    ArcCov = np.count_nonzero(Arc) / 100.0
    d = np.sqrt(U**2 + V**2) - Rad0
    mad = np.mean(np.abs(d))
    cir = {
        'radius': Rad0,
        'point': Point.flatten(),
        'mad': mad,
        'ArcCov': ArcCov,
        'conv': conv,
        'rel': rel
    }
    return cir


def least_squares_cylinder(P, cyl0, weight=None, Q=None):
    """
    Least-squares cylinder fitting using Gauss–Newton iterations.

    Input:
        P (np.Array): (n x 3) array representing the full point cloud.
        cyl0 (dictionary): Dictionary with initial cylinder parameters. Expected fields:
                - 'axis'   : initial axis direction (1 x 3)
                - 'start'  : initial axis point (1 x 3)
                - 'radius' : initial radius estimate
        weight (np.Array): Optional (n,) array of weights for the points.
        Q (np.Array): Optional (m x 3) subset of P where the cylinder is intended.

    Returns:
        dictionary  : Dictionary with the following fields:
                - 'radius'  : Fitted cylinder radius.
                - 'length'  : Cylinder length.
                - 'start'   : Axis point at the bottom of the cylinder (1 x 3).
                - 'axis'    : Fitted cylinder axis (1 x 3, unit vector).
                - 'mad'     : Mean absolute distance of points from the cylinder surface.
                - 'SurfCov' : Surface coverage of the cylinder.
                - 'dist'    : Radial distances from the points to the cylinder.
                - 'conv'    : True if the algorithm converged.
                - 'rel'     : True if the system was well conditioned.
    """
    res = 0.03  # Resolution level for computing surface coverage
    maxiter = 50
    iter_count = 0
    conv = False
    rel = True
    NoWeights = (weight is None)

    # Transform the data to nearly standard position.
    # Rot0 rotates cyl0.axis to the positive z-axis.
    Rot0, _, _ = rotate_to_z_axis(cyl0['axis'])
    Pt = (P - np.array(cyl0['start'])) @ Rot0.T
    np.round(Pt, decimals=8, out=Pt)
    # Initial estimates: translation and rotation angles are zero; radius is from cyl0.
    par = np.array([0, 0, 0, 0, cyl0['radius']], dtype=float)


    max_rad = np.max(np.max(P,axis = 0)-np.min(P,axis = 0))
    best_par = par
    # Gauss–Newton iterations to fit rotation-translation and radius parameters
    best_err = float("inf")
    while iter_count < maxiter and (not conv) and rel :
        if NoWeights:
            d0, J = func_grad_cylinder(par, Pt)
        else:
            d0, J = func_grad_cylinder(par, Pt, weight)
        SS0 = np.linalg.norm(d0)
        A = J.T @ J
        b = J.T @ d0
        try:
            p_update = -np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            rel = False
            if NoWeights:
                dist_new, _ = func_grad_cylinder(par, Pt)
            else:
                dist_new, _ = func_grad_cylinder(par, Pt, weight)
            break
        par = par + p_update

        # Check convergence: compute new distances.
        if NoWeights:
            dist_new, _ = func_grad_cylinder(par, Pt)
        else:
            dist_new, _ = func_grad_cylinder(par, Pt, weight)
        SS1 = np.linalg.norm(dist_new)
        err = abs(SS0 - SS1)
        if  err < 1e-4:
            conv = True
        if err < best_err:
            best_err = err
            best_par = par
        # Check reliability via the condition number of A.
        if np.linalg.cond(-A) != 0 and (1.0 / np.linalg.cond(-A)) < 10000 * np.finfo(float).eps:
            rel = False
        iter_count += 1

    if par[4]>max_rad or par[4]<.001:
        par = best_par
    # Compute final cylinder parameters.
    cyl_out = {}
    cyl_out['radius'] = float(par[4])

    # Inverse transformation: retrieve axis and axis point in original coordinates.
    # Form rotation matrix from fitted angles (par[2:4])
    R_fit, _, _ = form_rotation_matrices(par[2:4])
    # The fitted axis in the transformed space is [0,0,1]. Map it back:
    Axis = Rot0.T @ R_fit.T @ np.array([0, 0, 1])
    # The translation in the transformed space is given by par[0:2] (with a zero third component).
    Point_trans = np.array([par[0], par[1], 0])
    Point = Rot0.T @ Point_trans + np.array(cyl0['start'])

    # If Q is given and has more than 5 points, use Q instead of P for computing start, length, and mad.
    if Q is not None and Q.shape[0] > 5:
        P = Q
    else:
        P = P

    # Compute heights along the axis.
    H = P @ Axis
    hmin = np.min(H)
    cyl_length = float(np.abs(np.max(H) - hmin))
    # Translate the axis point to the bottom of the cylinder.
    hpoint = np.dot(Axis, Point)
    Point_adjusted = Point - (hpoint - hmin) * Axis

    cyl_out['start'] = Point_adjusted.flatten()
    cyl_out['axis'] = Axis.flatten()
    cyl_out['length'] = cyl_length



    mad = np.mean(np.abs(dist_new))
    cyl_out['mad'] = float(mad)

    cyl_out['conv'] = conv
    cyl_out['rel'] = rel

    # Compute surface coverage if Axis and Point are valid and the fit is reliable.
    if (not np.any(np.isnan(Axis))) and (not np.any(np.isnan(Point_adjusted))) and rel and conv:
        nl = max(3, int(np.ceil(cyl_length / res)))
        ns = int(np.ceil(2 * np.pi * cyl_out['radius'] / res))
        ns = min(36, max(ns, 8))
        # Assume Utils.surface_coverage returns a tuple; we take its first element.
        #SurfCov, _, _, _ = Utils.surface_coverage(P, Axis.reshape(1, -1), Point_adjusted.reshape(1, -1), nl, ns, 0.8 * cyl_out['radius'])
        SurfCov, _, _, _ = Utils.surface_coverage(P, Axis, Point_adjusted, nl, ns,
                                                    0.8 * cyl_out['radius'])
        cyl_out['SurfCov'] = float(SurfCov)
    else:
        cyl_out['SurfCov'] = 0.0

    return cyl_out