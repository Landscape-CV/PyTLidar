"""
Python adaptation and extension of TREEQSM.

Version: 0.0.4
Date: 19 March 2025
Authors: Fan Yang, John Hagood, Amir Hossein Alikhah Mishamandani
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PlottingUtils:
    @staticmethod
    def point_cloud_plotting(P, fig, ms=None, Bal=None, Sub=None):
        """
        Re-implementation of the MATLAB function:
            point_cloud_plotting(P, fig, ms, Bal, Sub)

        Parameters:
        -----------
        P   : numpy.ndarray
              The point cloud, shape can be (N,2) for 2D or (N,3) for 3D.
        fig : int
              Figure number (to mimic MATLAB's figure numbering).
        ms  : int, optional
              Marker size (if None or 0, defaults to 3).
        Bal : list of arrays/lists, optional
              Each element should be a list (or array) of point indices.
        Sub : list or list of lists, optional
              Either a list of indices or nested lists of indices referencing Bal.

        Additional Functionality:
        -------------------------
        Automatically saves the resulting plot as "point_cloud_plotting_fig_<fig>.png".

        Notes:
        ------
        - If only P and fig are provided (i.e., ms, Bal, Sub are None),
          it plots the entire point cloud in 2D or 3D.
        - If Bal is provided but Sub is None, it plots only the points
          specified by the union of all indices in Bal.
        - If both Bal and Sub are provided, it plots only the points
          specified by Bal[Sub], flattening Sub if it is nested.
        """

        # 1) Handle marker size logic (MATLAB style):
        #    If ms is None or 0, set default ms=3
        if ms is None or ms == 0:
            ms = 3

        # Prepare the figure
        plt.figure(num=fig)

        # 2) Decide whether to plot the entire point cloud or specific subsets:
        if Bal is None and Sub is None:
            # Equivalent to "if nargin < 4" in MATLAB
            # Plot all of P (either 2D or 3D)
            if P.shape[1] == 3:
                ax = plt.axes(projection='3d')
                ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=ms, c='b', marker='.')
            elif P.shape[1] == 2:
                plt.scatter(P[:, 0], P[:, 1], s=ms, c='b', marker='.')

        elif Bal is not None and Sub is None:
            # Equivalent to "if nargin == 4" in MATLAB
            # Flatten all indices in Bal
            all_indices = np.concatenate(Bal)
            ax = plt.axes(projection='3d')
            ax.scatter(P[all_indices, 0],
                       P[all_indices, 1],
                       P[all_indices, 2],
                       s=ms, c='b', marker='.')

        else:
            # Equivalent to "if nargin == 5" in MATLAB
            # Sub can be nested (lists of lists). We flatten accordingly.
            if isinstance(Sub, list):
                # Flatten Sub if it has deeper nesting
                #  - First flatten top-level if Sub is a list of lists
                flattened_sub = []
                for item in Sub:
                    if isinstance(item, list):
                        flattened_sub.extend(item)
                    else:
                        flattened_sub.append(item)
                # At this point flattened_sub is a simple list of indices (or still partial lists)
                #  - Flatten further if we still have lists inside
                final_sub = []
                for val in flattened_sub:
                    if isinstance(val, list):
                        final_sub.extend(val)
                    else:
                        final_sub.append(val)

                # Now gather indices from Bal based on final_sub
                all_indices = np.concatenate([Bal[idx] for idx in final_sub])
            else:
                # If Sub is just a single index or something else
                all_indices = np.concatenate(Bal[Sub])

            ax = plt.axes(projection='3d')
            ax.scatter(P[all_indices, 0],
                       P[all_indices, 1],
                       P[all_indices, 2],
                       s=ms, c='b', marker='.')

        # Try to set equal aspect (mostly works for 2D; for 3D it's approximate)
        if P.shape[1] == 2:
            plt.gca().set_aspect('equal', adjustable='box')
        # For 3D, exact 'axis equal' is trickier; this is a simple placeholder:
        else:
            plt.gca().set_box_aspect((1, 1, 1))

        # Show and save the figure
        plt.title(f"Figure {fig}")
        plt.savefig(f"point_cloud_plotting_fig_{fig}.png")
        plt.show()

    @staticmethod
    def plot_triangulation(QSM, fig=None, nf=None, AllTree=None):
        """
        Re-implementation of the MATLAB function:
            plot_triangulation(QSM, fig, nf, AllTree)

        Parameters
        ----------
        QSM      : object or dict-like
                   Should contain:
                     QSM.triangulation.vert   -> array of vertices
                     QSM.triangulation.facet  -> array of triangular facets
                     QSM.triangulation.cylind -> index of the first cylinder
                     QSM.triangulation.fvd    -> face/vertex color data
                     QSM.cylinder.branch      -> array to identify branch cylinders
        fig      : int, optional
                   Figure number. Default = 1 if not provided.
        nf       : int, optional
                   Number of facets for the cylinders. Default = 20.
        AllTree  : int, optional
                   If 1, plots the entire tree; otherwise only up to the last branch.
                   Default = 0.

        Additional:
        -----------
        - Automatically saves the figure as "plot_triangulation_fig_<fig>.png"
        - Relies on an existing helper function "plot_cylinder_model" to draw cylinders.
        - For alpha(1) effect (fully opaque patch), sets alpha=1 on the patch collection.
        """

        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # ------------------------------
        # 1) Handle default arguments
        # ------------------------------
        if QSM is None:
            raise ValueError("QSM must not be None.")

        if fig is None:
            fig = 1
        if nf is None:
            nf = 20
        if AllTree is None:
            AllTree = 0

        # ------------------------------
        # 2) Extract data from QSM
        # ------------------------------
        Vert   = np.array(QSM.triangulation.vert, dtype=float)
        Facets = np.array(QSM.triangulation.facet, dtype=int)
        CylInd = QSM.triangulation.cylind
        fvd    = np.array(QSM.triangulation.fvd, dtype=float)
        Bran   = np.array(QSM.cylinder.branch, dtype=int)
        nc     = Bran.shape[0]

        # Indices of all branch cylinders
        # (i.e., those with 'branch == 1')
        branch_indices = np.where(Bran == 1)[0] + 1  # +1 if your logic is 1-based

        # ------------------------------
        # 3) Figure setup
        # ------------------------------
        plt.figure(num=fig)
        ax = plt.axes(projection='3d')

        # ------------------------------
        # 4) Create the patch
        # ------------------------------
        # Facets might be 1-based in MATLAB, so subtract 1 if needed for Python
        # Check if any facet indices are >= number of vertices:
        if Facets.min() >= 1:
            # Adjust if they're 1-based (assuming no 0 in facets)
            Facets -= 1

        # Build polygons for each face
        polygons = [Vert[face] for face in Facets]

        # We'll map FaceVertexCData if it matches the faces
        # If 'fvd' is per-face data, we can convert it to colors.
        # For simplicity, treat 'fvd' as a scalar per face that we map to a colormap:
        if fvd.ndim == 1 and fvd.size == Facets.shape[0]:
            import matplotlib.cm as cm
            norm = plt.Normalize(vmin=fvd.min(), vmax=fvd.max())
            face_colors = cm.viridis(norm(fvd))
        else:
            # Fallback single color if shapes don't align
            face_colors = 'cyan'

        collection = Poly3DCollection(polygons, facecolors=face_colors, edgecolor='none', alpha=1.0)
        ax.add_collection3d(collection)

        # ------------------------------
        # 5) Determine which cylinders to plot
        # ------------------------------
        if AllTree == 1:
            Ind = np.arange(CylInd, nc + 1)  # +1 if using Python 1-based style
        else:
            if len(branch_indices) == 0:
                # If there's no "branch == 1", default to end anyway
                Ind = np.arange(CylInd, nc + 1)
            else:
                Ind = np.arange(CylInd, branch_indices[-1] + 1)

        # ------------------------------
        # 6) Plot the cylinder model
        # ------------------------------
        # Assumes a helper method: plot_cylinder_model(cylinder, fig, nf, something, branch=Ind)
        # Replace with your own call if needed.
        PlottingUtils.plot_cylinder_model(QSM.cylinder, fig, nf, 1, branch=Ind)

        # ------------------------------
        # 7) Final touches
        # ------------------------------
        ax.set_box_aspect((1,1,1))
        ax.set_title(f"Triangulation Plot - Fig {fig}")
        plt.savefig(f"plot_triangulation_fig_{fig}.png")
        plt.show()

    @staticmethod
    def plot_tree_structure2(P, Bal, Segs, SChi, fig=None, ms=None, BO=None, segind=None):
        """
        Re-implementation of the MATLAB function:
            plot_tree_structure2(P, Bal, Segs, SChi, fig, ms, BO, segind)

        Description:
        ------------
        - Plots a branch-segmented tree point cloud, where each branching order
          is shown in a different color.
        - Colors cycle through a predefined matrix (similar to MATLAB’s `col`).
        - If BO = 0, all branching orders are plotted; otherwise, only up to BO.
        - The plotting starts from a given segment index segind.

        Parameters:
        -----------
        P       : numpy.ndarray
                  3D point cloud array of shape (N,3).
        Bal     : list
                  Cover sets (e.g., Bal = cover.bal).
        Segs    : list
                  Segments (e.g., Segs = segment.segments).
                  Might be nested (cell of cell in MATLAB terms).
        SChi    : list
                  Child segments (SChi = segment.ChildSegment).
        fig     : int, optional
                  Figure number. Default = 1.
        ms      : int, optional
                  Marker size. Default = 3.
        BO      : int, optional
                  How many branching orders to plot (0 means plot all). Default = 0.
        segind  : int, optional
                  The segment index from which to begin plotting. Default = 1
                  (i.e., the main trunk if your indexing is 1-based).

        Additional:
        -----------
        - Saves the figure as "plot_tree_structure2_fig_<fig>.png"
        - Uses a repeated color table 'col' to assign each branching order a distinct color.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # Just to ensure 3D is recognized

        # -----------------------------
        # 1) Handle defaults
        # -----------------------------
        if fig is None:
            fig = 1
        if ms is None or ms == 0:
            ms = 3
        if BO is None:
            BO = 0
        if segind is None:
            segind = 1  # Typically 1-based in MATLAB

        # Repeat color matrix as in the MATLAB code
        base_col = np.array([
            [0.00, 0.00, 1.00],
            [0.00, 0.50, 0.00],
            [1.00, 0.00, 0.00],
            [0.00, 0.75, 0.75],
            [0.75, 0.00, 0.75],
            [0.75, 0.75, 0.00],
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
            [0.95, 0.95, 0.00],
            [0.25, 0.25, 0.75],
            [0.75, 0.75, 0.75],
            [0.00, 1.00, 0.00],
            [0.76, 0.57, 0.17],
            [0.54, 0.63, 0.22],
            [0.34, 0.57, 0.92],
            [1.00, 0.10, 0.60],
            [0.88, 0.75, 0.73],
            [0.10, 0.49, 0.47],
            [0.66, 0.34, 0.65],
            [0.99, 0.41, 0.23]
        ])
        col = np.tile(base_col, (1000, 1))  # replicate row-wise 1000 times

        # -----------------------------
        # 2) Flatten/Prepare Segs if needed
        #    This mirrors the MATLAB check "if iscell(Segs{1})"
        # -----------------------------
        # Check if the first element in Segs is itself a list of lists
        # (rough approximation of MATLAB iscell(...) logic)
        def is_nested_list(x):
            return isinstance(x, list) and any(isinstance(elem, list) for elem in x)

        if is_nested_list(Segs[0]):
            # Example: Segs is a list of N elements; each element is a list of sub-elements
            # Flatten each sub-list
            flattened_Segs = []
            for segset in Segs:
                # segset is, e.g., something like [[array(...)], [array(...)]...]
                S_accum = []
                for item in segset:
                    # item might be e.g. [array_of_indices]
                    # so item[0] is the actual array
                    if isinstance(item, list) and len(item) > 0:
                        # Typically item = [some_array]
                        # Flatten that array
                        S_accum.extend(item[0])
                    else:
                        # fallback scenario, direct or empty
                        S_accum.extend(item)
                flattened_Segs.append(np.array(S_accum, dtype=int))
            Seg = flattened_Segs
        else:
            # Otherwise Segs is already "flattened"
            Seg = Segs

        # -----------------------------
        # 3) If BO == 0, plot all branching orders
        # -----------------------------
        if BO == 0:
            BO = 1000

        # -----------------------------
        # 4) Start plotting
        # -----------------------------
        plt.figure(num=fig)
        ax = plt.axes(projection='3d')

        # The main set of indices for the starting segment
        # segind is likely 1-based from MATLAB, so shift if your data is 0-based
        # But we assume data is also 1-based in Python for direct parity:
        start_indices = Seg[segind - 1] if segind - 1 < len(Seg) else []
        # Flatten the Bal sets for these indices
        # (In MATLAB: S = vertcat(Bal{Seg{segind}});)
        # We'll do pythonic approach:
        S_list = []
        for idx in start_indices:
            S_list.extend(Bal[idx - 1])  # again, adjusting if Bal is also 1-based
        S = np.array(S_list, dtype=int)

        # Plot the trunk (or the main segment)
        ax.scatter(P[S, 0], P[S, 1], P[S, 2],
                   marker='.', s=ms, color=col[0, :])
        ax.set_box_aspect((1,1,1))

        # Keep track of all plotted points, to avoid re-plotting duplicates
        forb = set(S)  # forb = S in MATLAB

        # -----------------------------
        # 5) Recursively plot child branches, up to BO
        # -----------------------------
        if BO > 1:
            # In MATLAB: c = SChi{segind};
            if segind - 1 < len(SChi):
                c = SChi[segind - 1]  # children of the starting segment
            else:
                c = []

            i = 2
            while (i <= BO) and (len(c) > 0):
                # Unique, flattened list of child segment indices
                c_unique = np.unique(c)
                # Now get the Bal sets for all child segments
                child_indices = []
                for idxC in c_unique:
                    if idxC - 1 < len(Seg):
                        child_indices.extend(Seg[idxC - 1])
                child_indices_unique = np.unique(child_indices)

                # Flatten all Bal sets belonging to these child segments
                C_list = []
                for idx in child_indices_unique:
                    # idx is 1-based
                    if idx - 1 < len(Bal):
                        C_list.extend(Bal[idx - 1])
                C = np.array(C_list, dtype=int)

                # Exclude points we've already plotted
                C_diff = np.setdiff1d(C, list(forb))
                if C_diff.size > 0:
                    ax.scatter(P[C_diff, 0], P[C_diff, 1], P[C_diff, 2],
                               marker='.', s=ms, color=col[i-1, :])  # i-1 for zero-based indexing
                    ax.set_box_aspect((1,1,1))

                # Union of forb and these newly plotted points
                forb = np.union1d(forb, C_diff)

                # Next wave of children: c = unique(vertcat(SChi{c}))
                new_children = []
                for child_seg_index in c_unique:
                    if child_seg_index - 1 < len(SChi):
                        new_children.extend(SChi[child_seg_index - 1])
                c = new_children

                i += 1

        plt.title(f"Plot Tree Structure - Fig {fig}")
        plt.savefig(f"plot_tree_structure2_fig_{fig}.png")
        plt.show()

    @staticmethod
    def plot_tree_structure(P, cover, segment, fig=None, ms=None, segind=None, BO=None):
        """
        Python re-implementation of the MATLAB function plot_tree_structure(P, cover, segment, fig, ms, segind, BO).

        Description:
        ------------
        - Plots a branch-segmented point cloud with a distinct color for each branching order:
          Blue (stem), Green (1st-order), Red (2nd-order), etc.
        - If segind=1 and BO=0, only the stem is plotted.
          If segind=1 and BO=1, the stem and 1st-order branches are plotted.
          If segind=1 and BO >= max order (or BO not given), the entire tree is plotted.
          If segind=2 and BO is high enough/not given, branch #2 and all its sub-branches are plotted.
        - This function saves the resulting figure as "plot_tree_structure_fig_<fig>.png".

        Parameters:
        -----------
        P       : numpy.ndarray
                  Nx3 array of points (the tree point cloud).
        cover   : object or dict-like
                  Must contain 'cover.ball' (list of arrays), e.g. cover.ball => Bal.
        segment : object or dict-like
                  Must contain 'segment.segments' and 'segment.ChildSegment'.
        fig     : int, optional
                  Figure number. Defaults to 1 if not provided.
        ms      : int, optional
                  Marker size. Defaults to 1 if not provided.
        segind  : int, optional
                  Index of the segment to start plotting from. Defaults to 1 if not provided.
        BO      : int, optional
                  How many branching orders to plot. 0 = stem only.
                  If not provided or set to a large number, plots the entire subtree.

        Notes:
        ------
        - In MATLAB, indexing is 1-based. We assume the data in Python is also 1-based for
          direct parity. Adjust indices accordingly if your actual data is 0-based.
        - If 'segment.segments{1}' is itself a cell in MATLAB, we flatten it here.
        """

        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # ensures 3D projection is recognized

        # 1) Handle default arguments (mirroring MATLAB logic)
        if BO is None:
            BO = 1000  # large default so effectively "all orders"
        if segind is None:
            segind = 1
        if ms is None:
            ms = 1
        if fig is None:
            fig = 1

        Bal = cover.ball
        Segs = segment.segments
        SChi = segment.ChildSegment

        # 2) Define / repeat color matrix as in the MATLAB code
        base_col = np.array([
            [0.00, 0.00, 1.00],
            [0.00, 0.50, 0.00],
            [1.00, 0.00, 0.00],
            [0.00, 0.75, 0.75],
            [0.75, 0.00, 0.75],
            [0.75, 0.75, 0.00],
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
            [0.95, 0.95, 0.00],
            [0.25, 0.25, 0.75],
            [0.75, 0.75, 0.75],
            [0.00, 1.00, 0.00],
            [0.76, 0.57, 0.17],
            [0.54, 0.63, 0.22],
            [0.34, 0.57, 0.92],
            [1.00, 0.10, 0.60],
            [0.88, 0.75, 0.73],
            [0.10, 0.49, 0.47],
            [0.66, 0.34, 0.65],
            [0.99, 0.41, 0.23]
        ])
        col = np.tile(base_col, (1000, 1))  # replicate to ensure enough colors

        # 3) Flatten/prepare segments if needed (check if Segs[0] is itself a list)
        def is_nested_list(x):
            return isinstance(x, list) and any(isinstance(el, list) for el in x)

        if is_nested_list(Segs[0]):
            # Flatten each cell of Segs
            flattened = []
            for cell_item in Segs:
                S_accum = []
                for element in cell_item:
                    # 'element' may be something like [some_array]
                    # Flatten that array if needed
                    if isinstance(element, list):
                        S_accum.extend(element[0])  # typically element = [array]
                    else:
                        S_accum.extend(element)
                flattened.append(np.array(S_accum, dtype=int))
            Seg = flattened
        else:
            Seg = Segs

        # 4) Plot the starting segment
        plt.figure(num=fig)
        ax = plt.axes(projection='3d')

        # segind is 1-based, so we adjust if Python lists are 0-based
        start_index = segind - 1
        if start_index < 0 or start_index >= len(Seg):
            raise IndexError("segind is out of range for the 'segment.segments' data.")

        # Gather all ball-indices in the starting segment
        main_indices = Seg[start_index]
        # Flatten Bal sets for these indices
        main_points = []
        for idx in main_indices:
            # idx is also 1-based
            if idx - 1 < 0 or idx - 1 >= len(Bal):
                continue
            main_points.extend(Bal[idx - 1])

        main_points = np.array(main_points, dtype=int)
        ax.scatter(P[main_points, 0], P[main_points, 1], P[main_points, 2],
                   s=ms, c=[col[0]], marker='.')

        ax.set_box_aspect((1, 1, 1))
        plt.title(f"Plot Tree Structure - Fig {fig}")

        # 5) If BO > 0, plot successive branching orders
        if BO > 0:
            # hold on in MATLAB means we keep plotting on the same figure
            if start_index < len(SChi):
                c = SChi[start_index]
            else:
                c = []
            order = 1

            while (order <= BO) and len(c) > 0:
                # Flatten child segments
                child_segment_indices = []
                for child_idx in c:
                    # child_idx is 1-based
                    if child_idx - 1 < len(Seg):
                        child_segment_indices.extend(Seg[child_idx - 1])
                child_segment_indices = np.unique(child_segment_indices)

                # Gather all points from Bal sets for these child segments
                C_points = []
                for idxC in child_segment_indices:
                    if idxC - 1 < len(Bal):
                        C_points.extend(Bal[idxC - 1])
                C_points = np.array(C_points, dtype=int)

                # Plot them in the next color
                ax.scatter(P[C_points, 0], P[C_points, 1], P[C_points, 2],
                           s=ms, c=[col[order]], marker='.')
                ax.set_box_aspect((1, 1, 1))

                # Advance to grandchildren
                new_c = []
                for child_idx in np.unique(c):
                    if child_idx - 1 < len(SChi):
                        new_c.extend(SChi[child_idx - 1])
                c = new_c

                order += 1

        # 6) Save and show
        plt.savefig(f"plot_tree_structure_fig_{fig}.png")
        plt.show()

    @staticmethod
    def plot_spreads(treedata, fig=None, lw=None, rel=None):
        """
        Python re-implementation of the MATLAB function:
            plot_spreads(treedata, fig, lw, rel)

        Description:
        ------------
        - Plots the spread values (treedata.spreads) as polar plots, with each row
          in a different color.
        - If 'rel' is set to 1, then spreads are normalized by the global maximum,
          i.e., the maximum spread is plotted as radius=1. Otherwise, actual spread
          values are used.
        - Saves the resulting figure as "plot_spreads_fig_<fig>.png".

        Parameters:
        -----------
        treedata : object or dict-like
                   Must contain 'treedata.spreads', a 2D array of shape (n, m).
        fig      : int, optional
                   Figure number (defaults to 1 if not provided).
        lw       : float, optional
                   Line width (defaults to 1 if not provided).
        rel      : int, optional
                   If 1, plots relative (normalized) spreads. If 0, plots actual spreads.
                   Defaults to 1 if not provided.

        Notes:
        ------
        - In MATLAB, 'figure(fig)' picks or creates a figure window. Here, we just
          specify figure number in plt.figure(num=fig).
        - The polar plot uses equal angular spacing from 0 to 2*pi for each row's data.
        - The color array is automatically generated in a gradient from blue to magenta,
          approximating the logic in the MATLAB code.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # -----------------------------
        # 1) Handle default arguments
        # -----------------------------
        if fig is None:
            fig = 1
        if lw is None:
            lw = 1
        if rel is None:
            rel = 1

        spreads = treedata.spreads  # shape (n, m)
        n, m = spreads.shape

        # Create color gradient (similar to MATLAB code)
        # col(:,1) from 0 to (n-1)/n, col(:,2)=0, col(:,3) from 1 down to 1/n in steps of -1/n
        col = np.zeros((n, 3))
        col[:, 0] = np.linspace(0, (n-1)/n, n)    # Red channel
        col[:, 2] = np.linspace(1, 1/n, n)        # Blue channel
        # Green channel remains 0

        # Global max, used if rel=1 for normalization
        dmax = spreads.max() if spreads.size > 0 else 1.0

        # -----------------------------
        # 2) Setup the polar figure
        # -----------------------------
        plt.figure(num=fig)
        ax = plt.subplot(111, polar=True)

        # -----------------------------
        # 3) Plot the first row (as in MATLAB, done separately before the loop)
        #    D = [spreads(0,end), spreads(0,:)]
        # -----------------------------
        if n > 0:
            first_row = spreads[0]
            D = np.concatenate(([first_row[-1]], first_row))  # prepend last value to close the loop
            # Angular steps from 0..2*pi in len(D) points
            angles = np.linspace(0, 2*np.pi, len(D))
            if rel == 1:
                ax.plot(angles, D/dmax, '-', color=col[0], linewidth=lw)
            else:
                ax.plot(angles, D, '-', color=col[0], linewidth=lw)

        # -----------------------------
        # 4) Plot the rest of the rows
        #    for i = 1:n => which in MATLAB means 1-based indexing.
        #    We'll do range(n) in Python, which is 0-based,
        #    but to replicate the logic we skip i=0 here (already plotted).
        # -----------------------------
        for i in range(n):
            # If you want to replicate MATLAB's loop for i=1:1:n,
            # that means the entire range including i=1 => 0-based => i=0 => the first row.
            # That re-plots the first row in MATLAB. We'll replicate exactly so that
            # it shows the same behavior: the first row is plotted again in the loop.
            row_data = spreads[i]
            D = np.concatenate(([row_data[-1]], row_data))
            angles = np.linspace(0, 2*np.pi, len(D))
            if rel == 1:
                ax.plot(angles, D/dmax, '-', color=col[i], linewidth=lw)
            else:
                ax.plot(angles, D, '-', color=col[i], linewidth=lw)

        # -----------------------------
        # 5) Set radial limits and finalize
        # -----------------------------
        if rel == 1:
            ax.set_ylim(0, 1)  # rlim([0 1]) in MATLAB
        else:
            ax.set_ylim(0, dmax)

        plt.title(f"Spreads (Fig {fig})")
        plt.savefig(f"plot_spreads_fig_{fig}.png")
        plt.show()

    @staticmethod
    def plot_segs(P, comps, fig, ms, Bal=None):
        """
        Python re-implementation of the MATLAB function:
            plot_segs(P, comps, fig, ms, Bal)

        Description:
        ------------
        - Plots the point cloud segments given in the list "comps".
        - If Bal is not provided (i.e., 4 input arguments total), "comps" is assumed
          to directly contain point indices.
        - If Bal is provided (5 input arguments total), "comps" is assumed to contain
          cover set indices, which must be translated via 'Bal' to get point indices.
        - The resulting plot is saved as "plot_segs_fig_<fig>.png".

        Parameters:
        -----------
        P    : numpy.ndarray
               Point cloud of shape (N,3).
        comps: list
               List of segments; each segment is a list/array of indices or cover-set indices.
        fig  : int
               Figure number.
        ms   : int
               Marker size.
        Bal  : list, optional
               If provided, each element of comps[i] references cover sets in Bal.

        Notes:
        ------
        - When comps[i] is a list of lists, this code flattens them so that comps[i]
          becomes a single list/array of indices.
        - The color scheme replicates the MATLAB logic: up to 20 distinct "named" colors
          for n < 100 segments, else random colors.
        """

        import numpy as np
        import matplotlib.pyplot as plt

        # -----------------------------
        # 1) Define color matrix for up to 20 segments
        # -----------------------------
        base_col = np.array([
            [0.00, 0.00, 1.00],
            [0.00, 0.50, 0.00],
            [1.00, 0.00, 0.00],
            [0.00, 0.75, 0.75],
            [0.75, 0.00, 0.75],
            [0.75, 0.75, 0.00],
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
            [0.95, 0.95, 0.00],
            [0.25, 0.25, 0.75],
            [0.75, 0.75, 0.75],
            [0.00, 1.00, 0.00],
            [0.76, 0.57, 0.17],
            [0.54, 0.63, 0.22],
            [0.34, 0.57, 0.92],
            [1.00, 0.10, 0.60],
            [0.88, 0.75, 0.73],
            [0.10, 0.49, 0.47],
            [0.66, 0.34, 0.65],
            [0.99, 0.41, 0.23]
        ])

        n = len(comps)
        if n < 100:
            # Extend the base_col if needed
            reps = int(np.ceil(n / 20))
            col = np.tile(base_col, (reps, 1))
        else:
            # Create random colors for large n
            rng = np.random.default_rng()
            col = rng.random((n, 3))

        # -----------------------------
        # 2) Flatten nested comps if needed
        #    (Mirrors the check: if iscell(S) in MATLAB)
        # -----------------------------
        def is_list_of_lists(x):
            return isinstance(x, list) and any(isinstance(e, list) for e in x)

        if comps and is_list_of_lists(comps[0]):
            # Flatten each item in comps
            for i in range(n):
                S = comps[i]
                if S:
                    # S might be a list of arrays or lists
                    flattened_indices = []
                    for item in S:
                        flattened_indices.extend(item)
                    comps[i] = flattened_indices
                else:
                    comps[i] = []
        # If comps[i] is still a list of lists, you can flatten further if needed

        # -----------------------------
        # 3) Begin plotting
        # -----------------------------
        plt.figure(num=fig)
        ax = plt.axes(projection='3d')

        # -----------------------------
        # 4) If Bal is None => comps is direct point indices
        # -----------------------------
        if Bal is None:
            # Plot each segment
            if comps:
                # Plot first segment
                c0 = comps[0]
                ax.scatter(P[c0, 0], P[c0, 1], P[c0, 2],
                           c=[col[0]], marker='.', s=ms)

            # Plot subsequent segments
            for i in range(1, n):
                cseg = comps[i]
                ax.scatter(P[cseg, 0], P[cseg, 1], P[cseg, 2],
                           c=[col[i]], marker='.', s=ms)

        else:
            # -----------------------------
            # 5) Bal is provided => comps contains cover set indices
            # -----------------------------
            # Keep track of already plotted points in a boolean mask
            np_points = P.shape[0]
            D = np.zeros(np_points, dtype=bool)

            # Plot the first segment
            if comps and len(comps[0]) > 0:
                c0_sets = comps[0]
                # Flatten the unique point indices from Bal
                c0_points = np.unique(np.concatenate([Bal[idx] for idx in c0_sets]))
                ax.scatter(P[c0_points, 0], P[c0_points, 1], P[c0_points, 2],
                           c=[col[0]], marker='.', s=ms)

                # Mark them as used
                D[c0_points] = True

            # Plot subsequent segments
            for i in range(1, n):
                if comps[i]:
                    c_sets = comps[i]
                    c_points = np.unique(np.concatenate([Bal[idx] for idx in c_sets]))
                    # Exclude points already used
                    new_points = c_points[~D[c_points]]
                    D[new_points] = True
                    ax.scatter(P[new_points, 0], P[new_points, 1], P[new_points, 2],
                               c=[col[i]], marker='.', s=ms)

        ax.set_box_aspect((1,1,1))
        plt.title(f"Segments Plot - Fig {fig}")
        plt.savefig(f"plot_segs_fig_{fig}.png")
        plt.show()

    @staticmethod
    def plot_segments(P, Bal, fig, ms, seg1, seg2=None, seg3=None, seg4=None, seg5=None):
        """
        Python re-implementation of the MATLAB function:
            plot_segments(P, Bal, fig, ms, seg1, seg2, seg3, seg4, seg5)

        Description:
        ------------
        - Plots up to 5 segments (or subsets) of a point cloud.
        - Each segment is defined by a list/array of cover-set indices (seg1..seg5),
          which are converted to actual point indices via Bal.
        - If the segments overlap, points in earlier segments take precedence
          (are removed from later segments).
        - Seg1 is plotted in blue, seg2 in red, seg3 in green, seg4 in cyan, seg5 in magenta.
        - The resulting figure is saved as "plot_segments_fig_<fig>.png".

        Parameters:
        -----------
        P    : np.ndarray
               N x 3 array of point coordinates.
        Bal  : list
               'Bal' is a list of arrays, each array storing indices of points
               belonging to one cover set.
        fig  : int
               Figure number.
        ms   : float
               Marker size.
        seg1 : list or array-like
               Cover-set indices for segment 1 (plotted in blue).
        seg2 : list or array-like, optional
               Cover-set indices for segment 2 (plotted in red).
        seg3 : list or array-like, optional
               Cover-set indices for segment 3 (plotted in green).
        seg4 : list or array-like, optional
               Cover-set indices for segment 4 (plotted in cyan).
        seg5 : list or array-like, optional
               Cover-set indices for segment 5 (plotted in magenta).

        Behavior:
        ---------
        - If only seg1 is provided, only that set is plotted in blue.
        - If seg2 is provided, seg2's points are plotted in red, but any overlap
          with seg1 is removed.
        - Similarly for seg3 (green), seg4 (cyan), and seg5 (magenta).
        """
        import numpy as np
        import matplotlib.pyplot as plt

        plt.figure(num=fig)
        ax = plt.axes(projection='3d')

        # Helper to convert a list of Bal indices to the unique set of actual points
        def get_points_from_seg(seg_indices):
            # seg_indices is a list/array of cover-set indices (1-based or 0-based).
            # We assume 1-based indexing as in MATLAB. Adjust if necessary.
            all_points = []
            for idx in seg_indices:
                all_points.extend(Bal[idx])  # If 0-based in your data, remove +1
            return np.unique(all_points)

        # 1) Segment 1 (blue)
        S1 = get_points_from_seg(seg1)
        ax.scatter(P[S1, 0], P[S1, 1], P[S1, 2],
                   c='b', marker='.', s=ms)

        # 2) Segment 2 (red), removing overlap with S1
        if seg2 is not None:
            S2 = get_points_from_seg(seg2)
            S2 = np.setdiff1d(S2, S1)
            ax.scatter(P[S2, 0], P[S2, 1], P[S2, 2],
                       c='r', marker='.', s=ms)

        # 3) Segment 3 (green), removing overlap with S1 and S2
        if seg3 is not None:
            S3 = get_points_from_seg(seg3)
            if seg2 is not None:
                S3 = np.setdiff1d(S3, S2)
            S3 = np.setdiff1d(S3, S1)
            ax.scatter(P[S3, 0], P[S3, 1], P[S3, 2],
                       c='g', marker='.', s=ms)

        # 4) Segment 4 (cyan), removing overlap with S1, S2, S3
        if seg4 is not None:
            S4 = get_points_from_seg(seg4)
            if seg2 is not None:
                S4 = np.setdiff1d(S4, S2)
            if seg3 is not None:
                S4 = np.setdiff1d(S4, S3)
            S4 = np.setdiff1d(S4, S1)
            ax.scatter(P[S4, 0], P[S4, 1], P[S4, 2],
                       c='c', marker='.', s=ms)

        # 5) Segment 5 (magenta), removing overlap with S1..S4
        if seg5 is not None:
            S5 = get_points_from_seg(seg5)
            if seg2 is not None:
                S5 = np.setdiff1d(S5, S2)
            if seg3 is not None:
                S5 = np.setdiff1d(S5, S3)
            if seg4 is not None:
                S5 = np.setdiff1d(S5, S4)
            S5 = np.setdiff1d(S5, S1)
            ax.scatter(P[S5, 0], P[S5, 1], P[S5, 2],
                       c='m', marker='.', s=ms)

        ax.set_box_aspect((1, 1, 1))
        plt.title(f"Segments Plot - Fig {fig}")
        plt.savefig(f"plot_segments_fig_{fig}.png")
        plt.show()

    @staticmethod
    def plot_scatter(P, C, fig, ms):
        import numpy as np
        import matplotlib.pyplot as plt

        # Helper function to mimic MATLAB 'normalize' in the example
        def normalize(arr):
            arr = np.array(arr, dtype=float)
            arr_min, arr_max = arr.min(), arr.max()
            if arr_min == arr_max:
                return np.zeros_like(arr)
            return (arr - arr_min) / (arr_max - arr_min)

        # Normalize color data (S = normalize(C) in MATLAB)
        S = normalize(C)

        plt.figure(num=fig)

        # Check dimension of P to decide 2D or 3D scatter
        if P.shape[1] == 3:
            ax = plt.axes(projection='3d')
            # Original MATLAB: scatter3(..., ms*ones(size(P,1),1), C, 'filled')
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=ms, c=C, marker='o')
        else:
            # Original MATLAB (uncommented): scatter(..., ms*S.*ones(size(P,1),1), C, 'filled')
            # means the marker size is scaled by the normalized values S
            plt.scatter(P[:, 0], P[:, 1], s=ms * S, c=C, marker='o')

        plt.axis('equal')

        # If C is a single column of values, add a colorbar
        if C.ndim == 1:
            # Set colormap and colorbar range
            plt.colormap = plt.cm.jet  # for Jupyter, or see below for typical usage
            cmin, cmax = C.min(), C.max()
            if cmin < cmax:
                norm = plt.Normalize(vmin=cmin, vmax=cmax)
            else:
                # If all C are the same, fallback to a dummy range
                norm = plt.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
            plt.colorbar(sm)

        plt.savefig(f"plot_scatter_fig_{fig}.png")
        plt.show()

    @staticmethod
    def plot_point_cloud(P, fig=None, ms=None, col=None):
        """
        Python re-implementation of the MATLAB function:
            plot_point_cloud(P, fig, ms, col)

        Description:
        ------------
        - Plots a 2D or 3D point cloud P.
        - Default figure number = 1, default marker size = 3, default color = 'b' (blue).
        - Saves the resulting figure as "plot_point_cloud_fig_<fig>.png".

        Parameters:
        -----------
        P   : np.ndarray
              2D (N,2) or 3D (N,3) array of coordinates.
        fig : int, optional
              Figure number (defaults to 1 if not provided).
        ms  : int, optional
              Marker size (defaults to 3 if not provided or if set to 0).
        col : str, optional
              Color specification (defaults to 'b').

        Notes:
        ------
        - For 3D data, uses plot3-like functionality via plt.axes(projection='3d').
        - For 2D data, uses standard 2D plot.
        """
        import matplotlib.pyplot as plt

        # 1) Handle default arguments
        if fig is None:
            fig = 1
        if ms is None or ms == 0:
            ms = 3
        if col is None:
            col = 'b'

        # 2) Prepare figure
        plt.figure(num=fig)

        # 3) Plot either in 3D or 2D
        if P.shape[1] == 3:
            ax = plt.axes(projection='3d')
            ax.plot3D(P[:, 0], P[:, 1], P[:, 2],
                      f'.{col}', markersize=ms)
        else:
            plt.plot(P[:, 0], P[:, 1],
                     f'.{col}', markersize=ms)

        plt.axis('equal')
        plt.title(f"Point Cloud - Fig {fig}")

        # 4) Save and show
        plt.savefig(f"plot_point_cloud_fig_{fig}.png")
        plt.show()

    @staticmethod
    def plot_branch_segmentation(P, cover, segment, Color='order', fig=1, ms=1, segind=1, BO=1000, ax = None):
        """
        Python re-implementation of MATLAB's plot_branch_segmentation.

        Plots a branch‐segmented point cloud, coloring either by unique branch or
        by branching order. In 'branch' mode each segment gets a unique random color
        (ensuring a minimum difference from its parent's color). In 'order' mode a
        predefined colormap is used: Blue = trunk, Green = 1st-order, Red = 2nd-order, etc.

        Parameters
        ----------
        P       : np.ndarray
                  (N x 3) array representing the point cloud.
        cover   : dict-like
                  Must contain key "ball" (e.g., cover["ball"]) holding a list of cover-set
                  point index arrays (assumed 1-indexed).
        segment : dict-like
                  Must contain:
                    - "segments": a list where each element is either an array of cover-set
                      indices or a nested list that needs flattening.
                    - "ChildSegment": list of child segments for each segment (1-indexed).
                    - "ParentSegment": list of parent segment indices (1-indexed).
        Color   : str, optional
                  'order' (default) or 'branch'. If 'order', colors are chosen from a fixed
                  colormap; if 'branch', each segment is assigned a unique random color.
        fig     : int, optional
                  Figure number (default 1).
        ms      : int, optional
                  Marker size (default 1).
        segind  : int, optional
                  Starting segment index (1-indexed, default 1).
        BO      : int, optional
                  Number of branching orders to plot (default 1000, i.e. plot entire tree).
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # Extract required fields from cover and segment structures.
        Bal = np.array(cover["ball"], dtype = 'object')
        Segs = segment["segments"]
        SChi = segment["ChildSegment"]
        SPar = segment["ParentSegment"]
        ns_total = len(Segs)

        # Flatten Segs if needed: if the first element is itself a list.
        if isinstance(Segs[0], list) and any(isinstance(item, list) for item in Segs[0]):
            Seg = []
            for seg_cell in Segs:
                flat_seg = []
                for item in seg_cell:
                    # Assume each item is a list/array of cover indices.
                    flat_seg.extend(item if isinstance(item, list) else list(item))
                Seg.append(np.array(flat_seg, dtype=int))
        else:
            # Otherwise assume Segs is already a list of arrays.
            Seg = Segs

        # Set up color mode.
        if Color.lower() == 'branch':
            mode = 1  # branch mode: assign each segment a unique random color.
            # Create an array of random colors (one per segment).
            col = np.random.rand(ns_total, 3)
            # Adjust colors for segments (starting from the second) so that each differs
            # from its parent's color by a minimum threshold.
            for i in range(1, ns_total):
                parent_idx = int(SPar[i]) # adjust for 0-based indexing
                parent_color = col[parent_idx]
                current_color = col[i]
                while np.sum(np.abs(parent_color - current_color)) < 0.2:
                    current_color = np.random.rand(3)
                col[i] = current_color
        elif Color.lower() == 'order':
            mode = 0  # order mode: use fixed colormap.
            base_col = np.array([
                [0.00, 0.00, 1.00],
                [0.00, 0.50, 0.00],
                [1.00, 0.00, 0.00],
                [0.00, 0.75, 0.75],
                [0.75, 0.00, 0.75],
                [0.75, 0.75, 0.00],
                [0.25, 0.25, 0.25],
                [0.75, 0.25, 0.25],
                [0.95, 0.95, 0.00],
                [0.25, 0.25, 0.75],
                [0.75, 0.75, 0.75],
                [0.00, 1.00, 0.00],
                [0.76, 0.57, 0.17],
                [0.54, 0.63, 0.22],
                [0.34, 0.57, 0.92],
                [1.00, 0.10, 0.60],
                [0.88, 0.75, 0.73],
                [0.10, 0.49, 0.47],
                [0.66, 0.34, 0.65],
                [0.99, 0.41, 0.23]
            ])
            col = np.tile(base_col, (10, 1))
        else:
            # Default to 'order' if unrecognized.
            mode = 0
            base_col = np.array([
                [0.00, 0.00, 1.00],
                [0.00, 0.50, 0.00],
                [1.00, 0.00, 0.00],
                [0.00, 0.75, 0.75],
                [0.75, 0.00, 0.75],
                [0.75, 0.75, 0.00],
                [0.25, 0.25, 0.25],
                [0.75, 0.25, 0.25],
                [0.95, 0.95, 0.00],
                [0.25, 0.25, 0.75],
                [0.75, 0.75, 0.75],
                [0.00, 1.00, 0.00],
                [0.76, 0.57, 0.17],
                [0.54, 0.63, 0.22],
                [0.34, 0.57, 0.92],
                [1.00, 0.10, 0.60],
                [0.88, 0.75, 0.73],
                [0.10, 0.49, 0.47],
                [0.66, 0.34, 0.65],
                [0.99, 0.41, 0.23]
            ])
            col = np.tile(base_col, (10, 1))

        # Build the list of segments to plot and record their branching order.
        segments = [segind]  # start with the initial segment (1-indexed)
        order_list = [1]
        # Get child segments for the starting segment (adjust for 0-based indexing)
        C = SChi[int(segind)]
        b = 0
        while len(C)>0 and (b <= BO):
            b += 1
            segments.extend(C)
            order_list.extend(np.array([b] * len(C)))
            # For each segment in C, get its children and flatten.
            new_C = []
            for idx in C:
                new_C.extend(SChi[int(idx)])
            C = new_C

        ns_segments = len(segments)

        # Create figure and 3D axis.
        # fig_obj = plt.figure(num=fig)
        if ax is None:
            ax = plt.axes(projection='3d')

        # For each segment in the list, collect the corresponding point indices and plot.
        for i in range(ns_segments):
            seg_idx = segments[i]
            # Get cover-set indices from the flattened Seg for this segment.
            seg_cover_indices = Seg[int(seg_idx)]
            pts = []
            for cover_idx in seg_cover_indices:
                pts.extend(Bal[cover_idx.astype(int)])  # Convert to int if needed
            pts = np.concatenate([pt for pt in pts])
            if pts.size > 0:
                if mode == 1:  # branch mode: color by segment index.
                    color = col[int(seg_idx)]
                else:  # order mode: color by branching order.
                    color = col[order_list[i]]
                ax.scatter(P[pts, 0], P[pts, 1], P[pts, 2],
                           c=[color], marker='.', s=ms)
        ax.set_box_aspect((1, 1, 1))
        plt.title(f"Branch Segmentation (Fig {fig})")
        # plt.show()

    @staticmethod
    def plot_models_segmentations(P, cover, segment, cylinder, trunk=None, triangulation=None):
        """
        Python re-implementation of the MATLAB function:
            plot_models_segmentations(P, cover, segment, cylinder, trunk, triangulation)

        Description:
        ------------
        - Plots several figures showing:
          (1) Branch-segmented point cloud with colors denoting branching order vs. branches
          (2) Cylinder model with colors denoting branching order vs. branches
          (3) Combined branch-segmented cloud and cylinder model in one figure
          (4) (Optional) Triangulation model (bottom) plus cylinder model (top) of the stem

        Parameters:
        -----------
        P            : np.ndarray
                       N x 3 array of point coordinates (the main point cloud).
        cover        : dict-like
                       Must have cover["ball"] or similar needed by the called
                       plot functions (e.g. plot_branch_segmentation).
        segment      : dict-like
                       Must have necessary fields for branch segmentation (e.g., segment["segments"]).
        cylinder     : dict-like
                       Must have 'start', 'branch', etc., used by plot_cylinder_model().
        trunk        : np.ndarray or None, optional
                       The trunk points (N x 3). Used in figure 4 if provided.
        triangulation: dict-like or None, optional
                       Should have triangulation["vert"], triangulation["facet"],
                       triangulation["fvd"], triangulation["cylind"] if provided
                       (for figure 4).

        Notes / Dependencies:
        ---------------------
        - This function references other helper functions:
            plot_branch_segmentation(...)
            plot_cylinder_model(...)
            point_cloud_plotting(...)
          which should be defined separately or imported from your codebase.
        - The logic shifts the entire cloud (and trunk, etc.) by the first row of
          cylinder.start so that the base is near the origin, mirroring the MATLAB code.
        - The subplots, figure numbering, and alpha settings replicate the MATLAB flow.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # 1) Figure 1: Branch-segmented point cloud
        #    Left subplot: colors = branching order
        #    Right subplot: colors = branch
        plt.figure(1, figsize=(10, 5))
        plt.clf()

        ax = plt.subplot(1, 2, 1, projection='3d')
        # (Assumes a helper function like: plot_branch_segmentation(P, cover, segment, mode, fig=None, subplot=None))
        # 'order' color mode:
        PlottingUtils.plot_branch_segmentation(P, cover, segment, Color='order', ax = ax)
        plt.title("Branch segmentation (order)")

        ax = plt.subplot(1, 2, 2, projection='3d')
        # 'branch' color mode:
        PlottingUtils.plot_branch_segmentation(P, cover, segment, Color='branch', ax = ax)
        plt.title("Branch segmentation (branch)")

        plt.tight_layout()
        plt.show()

        # 2) Figure 2: Cylinder model
        #    The cylinder is also colored by 'order' (subplot 1) vs. 'branch' (subplot 2).
        #    The code shifts P by Sta(1,:) => in Python, Sta[0] if zero-based.
        if "start" in cylinder:
            Sta = np.array(cylinder["start"])  # Expect shape (N,3); the code specifically uses Sta[1,:] in MATLAB
            if Sta.ndim == 2 and Sta.shape[0] > 0:
                shift_vec = Sta[0]
            else:
                shift_vec = np.zeros(3)
        else:
            shift_vec = np.zeros(3)

        # Shift the main cloud (P) by shift_vec
        P_shifted = P - shift_vec

        # If trunk, triangulation exist, shift them too
        if trunk is not None:
            trunk_shifted = trunk - shift_vec
        else:
            trunk_shifted = None

        Vert_shifted = None
        if triangulation is not None and "vert" in triangulation:
            Vert_shifted = np.array(triangulation["vert"], dtype=float) - shift_vec

        # Also shift cylinder["start"] so the model is consistent
        if "start" in cylinder:
            cylinder_start = np.array(cylinder["start"]) - shift_vec
            cylinder["start"] = cylinder_start

        # Show figure 2 with subplots
        plt.figure(2, figsize=(10, 5))
        plt.clf()

        # subplot(1,2,1) => cylinder with 'order' coloring, nf=10, radius factor=2
        ax = plt.subplot(1, 2, 1, projection='3d')
        PlottingUtils.plot_cylinder_model(cylinder, Color='order', fig=None, nf=10,ax = ax)
        plt.title("Cylinder model (order)")

        # subplot(1,2,2) => cylinder with 'branch' coloring, nf=10
        ax = plt.subplot(1, 2, 2, projection='3d')
        PlottingUtils.plot_cylinder_model(cylinder, Color='branch', fig=None, nf=10,ax = ax)
        plt.title("Cylinder model (branch)")

        plt.tight_layout()
        plt.show()

        # 3) Figure 3: combined segmented point cloud + cylinder model
        #    Both colored by 'order' in MATLAB. With partial transparency on the cylinders.
        plt.figure(3)
        plt.clf()

        # Plot branch segmentation in 'order' mode
        PlottingUtils.plot_branch_segmentation(P_shifted, cover, segment, Color='order', fig=None)
        # "hold on" => in matplotlib, we simply plot on the current axis
        plt.figure(4)
        plt.clf()
        PlottingUtils.plot_cylinder_model(cylinder, Color='order', fig=None, nf=10, alp=0.7)
        plt.title("Segmentation + Cylinder model (order)")
        plt.show()

        # 4) Figure 4: Triangulation model (bottom) and cylinder model (top) of the stem
        #    Only if trunk and triangulation are valid.
        #    The code also picks out trunk points randomly for illustration,
        #    then calls point_cloud_plotting, patch, alpha(1), hold on, etc.
        #    We also gather indices for the trunk portion of the cylinder with
        #    'branch==1'. Then plot the triangulation patch, then cylinders from
        #    index CylInd to the last trunk-based cylinder.
        if trunk_shifted is not None and Vert_shifted is not None:
            if Vert_shifted.shape[0] > 5:  # "max(size(Vert))>5" in MATLAB
                # Triangulation data
                Facets = triangulation["facet"]
                fvd    = triangulation["fvd"]
                CylInd = triangulation["cylind"]  # index of first trunk cylinder
                Bran   = cylinder["branch"]       # array of branch indicators
                nc     = len(Bran)
                # gather indices of trunk-based cylinders (branch==1)
                branch_indices = np.where(Bran == 1)[0] + 1  # if 1-based
                if len(branch_indices) > 0:
                    last_trunk = branch_indices[-1]
                else:
                    last_trunk = nc

                # Random subset of trunk points?
                # "I = logical(round(0.55*rand(n,1)))" => ~ 55% of trunk
                n_trunk = trunk_shifted.shape[0]
                import numpy as np
                rand_mask = np.random.rand(n_trunk)  # uniform in [0,1)
                I = rand_mask < 0.55

                plt.figure(4)
                plt.clf()

                # Plot random trunk points
                PlottingUtils.point_cloud_plotting(trunk_shifted[I, :], fig=4, ms=3)

                # Then the triangulation patch
                # In Python, we might do a 3D Poly3DCollection. For brevity,
                # let's just assume a "plot_triangulation_patch" helper
                # or replicate the user’s patch approach:
                PlottingUtils.plot_triangulation_patch(Vert_shifted, Facets, fvd, alpha_value=1.0)

                # "hold on" => just keep the same axis
                # Now plot cylinder from 'CylInd' to 'last_trunk':
                PlottingUtils.plot_cylinder_model(cylinder,
                                    color_mode='order',
                                    fig=4,
                                    nf=20,
                                    alp=1.0,
                                    subset=range(CylInd, last_trunk+1))
                plt.title("Triangulation + Cylinder (stem)")
                plt.axis('equal')
                plt.show()
            else:
                print("No triangulation model generated!")


    @staticmethod
    def plot_cylinder_model(cylinder, Color='order', fig=1, nf=20, alp=1, Ind=None, ax = None):
        """
        Re-implementation of MATLAB’s plot_cylinder_model.

        Parameters
        ----------
        cylinder : dict‐like
            Must contain fields "radius", "length", "start", "axis", "BranchOrder"
            and (if using branch‐coloring) "branch" (and optionally "parent").
        Color : str, optional
            'order' (default) colors using a fixed colormap; 'branch' assigns each branch a unique random color.
        fig : int, optional
            Figure number (default 1).
        nf : int, optional
            Number of facets in the thickest cylinder (default 20; scales down for thinner cylinders, min 4).
        alp : float, optional
            Alpha value (1 = opaque, 0 = fully transparent).
        Ind : array-like, optional
            If provided, only the cylinders with these indices are plotted.

        Notes
        -----
        This method builds a standard cylinder template (for a range of facet numbers),
        scales, rotates, and translates it for each cylinder in the model, then creates a
        patch (via Poly3DCollection) to simulate the smooth surface.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # Subset fields if Ind is provided
        if Ind is not None:
            Rad = np.array(cylinder["radius"])[Ind]
            Len = np.array(cylinder["length"])[Ind]
            Sta = np.array(cylinder["start"])[Ind, :]
            Axe = np.array(cylinder["axis"])[Ind, :]
            BOrd = np.array(cylinder["BranchOrder"])[Ind]
            if Color.lower() == 'branch':
                Bran = np.array(cylinder["branch"])[Ind]
        else:
            Rad = np.array(cylinder["radius"])
            Len = np.array(cylinder["length"])
            Sta = np.array(cylinder["start"])
            Axe = np.array(cylinder["axis"])
            BOrd = np.array(cylinder["BranchOrder"])
            if Color.lower() == 'branch':
                Bran = np.array(cylinder["branch"])
        nc = Rad.shape[0]

        # Set up colormap depending on the coloring mode.
        if Color.lower() == 'order':
            mode = 1
            base_col = np.array([
                [0.00, 0.00, 1.00],
                [0.00, 0.50, 0.00],
                [1.00, 0.00, 0.00],
                [0.00, 0.75, 0.75],
                [0.75, 0.00, 0.75],
                [0.75, 0.75, 0.00],
                [0.25, 0.25, 0.25],
                [0.75, 0.25, 0.25],
                [0.95, 0.95, 0.00],
                [0.25, 0.25, 0.75],
                [0.75, 0.75, 0.75],
                [0.00, 1.00, 0.00],
                [0.76, 0.57, 0.17],
                [0.54, 0.63, 0.22],
                [0.34, 0.57, 0.92],
                [1.00, 0.10, 0.60],
                [0.88, 0.75, 0.73],
                [0.10, 0.49, 0.47],
                [0.66, 0.34, 0.65],
                [0.99, 0.41, 0.23]
            ])
            col = np.tile(base_col, (10, 1))
        elif Color.lower() == 'branch':
            mode = 0
            N = int(np.max(Bran))
            col = np.random.rand(N+1, 3)
            # Ensure each branch’s color differs sufficiently from its parent's.
            parent = np.array(cylinder.get("parent", np.zeros(nc)))
            for i in range(1, nc):
                pidx = int(parent[i])
                C = col[Bran[pidx]]
                c = col[int(Bran[i])]
                while np.sum(np.abs(C- c)) < 0.2:
                    c = np.random.rand(3)
                col[int(Bran[i])] = c
        else:
            mode = 1
            col = np.tile(np.array([[0.00, 0.00, 1.00]]), (nc, 1))

        # Precompute standard cylinder templates for facet counts 4..nf.
        Cir = {}
        for i in range(4, nf + 1):
            theta = np.linspace(0, 2 * np.pi, i, endpoint=False)
            B = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(i)))
            T = np.column_stack((np.cos(theta), np.sin(theta), np.ones(i)))
            # Facet connectivity: each facet is defined by 4 vertices.
            facets = np.column_stack((
                np.arange(1, i + 1),
                np.arange(i + 1, 2 * i + 1),
                np.concatenate((np.arange(i + 2, 2 * i + 1), [i + 1])),
                np.concatenate((np.arange(2, i + 1), [1]))
            ))
            Cir[i] = (np.vstack((B, T)), facets)

        # Allocate arrays (overestimate size).
        max_vertices = 2 * nc * (nf + 1)
        Vert = np.zeros((max_vertices, 3))
        max_facets = nc * (nf + 1)
        Facets = np.zeros((max_facets, 4), dtype=int)
        fvd = np.zeros((max_facets, 3))
        t = 0
        f = 0

        # Helper functions.
        def rotation_matrix(axis, angle):
            axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) != 0 else axis
            a = np.cos(angle / 2)
            b, c, d = -axis * np.sin(angle / 2)
            return np.array([
                [a * a + b * b - c * c - d * d, 2 * (b * c - a * d),       2 * (b * d + a * c)],
                [2 * (b * c + a * d),       a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                [2 * (b * d - a * c),       2 * (c * d + a * b),       a * a + d * d - b * b - c * c]
            ])
        def mat_vec_subtraction(mat, vec):
            return mat - vec  # broadcasting subtraction

        # Loop over each cylinder.
        for i in range(nc):
            n_facets = int(np.ceil(np.sqrt(Rad[i] / Rad[0]) * nf))
            n_facets = min(n_facets, nf)
            n_facets = max(n_facets, 4)
            C = np.copy(Cir[n_facets][0])  # standard template vertices
            # Scale: bottom circle scaled by radius, top circle’s z scaled by length.
            C[:n_facets, :2] = Rad[i] * C[:n_facets, :2]
            C[n_facets:, 2] = Len[i] * C[n_facets:, 2]
            # Rotate: compute rotation to align z-axis with cylinder axis.
            ang = np.arccos(np.clip(Axe[i, 2], -1, 1))
            axis_rot = np.cross(np.array([0, 0, 1]), Axe[i])
            Rot = rotation_matrix(axis_rot, ang) if np.linalg.norm(axis_rot) >= 1e-6 else np.eye(3)
            C = (Rot @ C.T).T
            # Translate: add the starting position.
            C = mat_vec_subtraction(C, -Sta[i])
            n_vert = 2 * n_facets
            Vert[t:t + n_vert, :] = C
            Facets[f:f + n_facets, :] = Cir[n_facets][1] + t
            # Assign face colors based on branch order (or branch) – note the 1-offset adjustments.
            if mode == 1:
                fvd[f:f + n_facets, :] = np.tile(col[BOrd[i] + 1], (n_facets, 1))
            else:
                fvd[f:f + n_facets, :] = np.tile(col[Bran[i]], (n_facets, 1))
            t += n_vert
            f += n_facets

        # Trim unused preallocated space.
        Vert = Vert[:t, :]
        Facets = Facets[:f, :]
        fvd = fvd[:f, :]

        # Plot using a 3D patch.
        # fig_obj = plt.figure(fig)
        if ax is None:
            ax = plt.axes(projection='3d')
        ax.plot3D([Vert[0, 0]], [Vert[0, 1]], [Vert[0, 2]], 'o')
        poly = Poly3DCollection(Vert[Facets - 1], facecolors=fvd, edgecolor='none', alpha=alp)
        ax.add_collection3d(poly)
        ax.set_box_aspect((1, 1, 1))
        ax.grid(True)
        ax.view_init(elev=30, azim=-37.5)
        # plt.show()

    @staticmethod
    def plot_cone_model(cylinder, fig=1, nf=20, alp=1, Ind=None):
        """
        Re-implementation of MATLAB’s plot_cone_model as truncated cones.

        Parameters
        ----------
        cylinder : dict‐like
            Must contain fields "radius", "TopRadius", "length", "start", "axis",
            and (if needed) other branch info.
        fig : int, optional
            Figure number.
        nf : int, optional
            Number of facets (default 20, with scaling down to a minimum of 4).
        alp : float, optional
            Alpha (transparency) value.
        Ind : array-like, optional
            Indices to plot a subset of cylinders.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if isinstance(cylinder, dict):
            Rad = np.array(cylinder["radius"])
            Rad2 = np.array(cylinder["TopRadius"])
            Len = np.array(cylinder["length"])
            Sta = np.array(cylinder["start"])
            Sta = Sta - Sta[0]
            Axe = np.array(cylinder["axis"])
            BOrd = np.array(cylinder["BranchOrder"])
        else:
            # If cylinder is a numpy array (alternative format)
            Rad = cylinder[:, 0]
            Len = cylinder[:, 1]
            Sta = cylinder[:, 2:5]
            Sta = Sta - Sta[0]
            Axe = cylinder[:, 5:8]
            BOrd = cylinder[:, 13]
        if Ind is not None:
            Rad = Rad[Ind]
            Len = Len[Ind]
            Sta = Sta[Ind, :]
            Axe = Axe[Ind, :]
            BOrd = BOrd[Ind]
        nc = Rad.shape[0]

        # Choose colormap based on BranchOrder.
        base_col = np.array([
            [0.00, 0.00, 1.00],
            [0.00, 0.50, 0.00],
            [1.00, 0.00, 0.00],
            [0.00, 0.75, 0.75],
            [0.75, 0.00, 0.75],
            [0.75, 0.75, 0.00],
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
            [0.95, 0.95, 0.00],
            [0.25, 0.25, 0.75],
            [0.75, 0.75, 0.75],
            [0.00, 1.00, 0.00],
            [0.76, 0.57, 0.17],
            [0.54, 0.63, 0.22],
            [0.34, 0.57, 0.92],
            [1.00, 0.10, 0.60],
            [0.88, 0.75, 0.73],
            [0.10, 0.49, 0.47],
            [0.66, 0.34, 0.65],
            [0.99, 0.41, 0.23]
        ])
        N = int(np.max(BOrd)) + 1
        if N <= 20:
            col = base_col[:N, :]
        else:
            m = int(np.ceil(N / 20))
            col = np.tile(base_col, (m, 1))[:N, :]

        # Precompute standard templates.
        Cir = {}
        for i in range(4, nf + 1):
            theta = np.linspace(0, 2 * np.pi, i, endpoint=False)
            B = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(i)))
            T = np.column_stack((np.cos(theta), np.sin(theta), np.ones(i)))
            Cir[i] = (np.vstack((B, T)), np.column_stack((
                np.arange(1, i + 1),
                np.arange(i + 1, 2 * i + 1),
                np.concatenate((np.arange(i + 2, 2 * i + 1), [i + 1])),
                np.concatenate((np.arange(2, i + 1), [1]))
            )))

        max_vertices = 2 * nc * (nf + 1)
        Vert = np.zeros((max_vertices, 3))
        max_facets = nc * (nf + 1)
        Facets = np.zeros((max_facets, 4), dtype=int)
        t = 0
        f = 0

        def rotation_matrix(axis, angle):
            axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) != 0 else axis
            a = np.cos(angle / 2)
            b, c, d = -axis * np.sin(angle / 2)
            return np.array([
                [a * a + b * b - c * c - d * d, 2 * (b * c - a * d),       2 * (b * d + a * c)],
                [2 * (b * c + a * d),       a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                [2 * (b * d - a * c),       2 * (c * d + a * b),       a * a + d * d - b * b - c * c]
            ])
        def mat_vec_subtraction(mat, vec):
            return mat - vec

        Vert_all = np.zeros((max_vertices, 3))
        Facets_all = np.zeros((max_facets, 4), dtype=int)
        fvd = np.tile(np.array([139 / 255, 69 / 255, 19 / 255]), (max_facets, 1))
        t = 0
        f = 0
        for i in range(nc):
            n_facets = int(np.ceil(np.sqrt(Rad[i] / Rad[0]) * nf))
            n_facets = min(n_facets, nf)
            n_facets = max(n_facets, 4)
            C = np.copy(Cir[n_facets][0])
            m_total = C.shape[0]
            half = m_total // 2
            C[:half, :2] = Rad[i] * C[:half, :2]
            C[half:, :2] = Rad2[i] * C[half:, :2]
            C[n_facets:, 2] = Len[i] * C[n_facets:, 2]
            ang = np.arccos(np.clip(Axe[i, 2], -1, 1))
            axis_rot = np.cross(np.array([0, 0, 1]), Axe[i])
            Rot = rotation_matrix(axis_rot, ang) if np.linalg.norm(axis_rot) >= 1e-6 else np.eye(3)
            C = (Rot @ C.T).T
            C = mat_vec_subtraction(C, -Sta[i])
            n_vert = m_total
            Vert_all[t:t + n_vert, :] = C
            Facets_all[f:f + n_facets, :] = Cir[n_facets][1] + t
            fvd[f:f + n_facets, :] = np.tile(col[int(BOrd[i]) + 1], (n_facets, 1))
            t += n_vert
            f += n_facets
        Vert_all = Vert_all[:t, :]
        Facets_all = Facets_all[:f, :]
        fvd = fvd[:f, :]

        fig_obj = plt.figure(fig)
        ax = plt.axes(projection='3d')
        ax.plot3D([Vert_all[0, 0]], [Vert_all[0, 1]], [Vert_all[0, 2]], 'o')
        poly = Poly3DCollection(Vert_all[Facets_all - 1], facecolors=fvd, edgecolor='none', alpha=alp)
        ax.add_collection3d(poly)
        ax.set_box_aspect((1, 1, 1))
        ax.grid(True)
        ax.view_init(elev=30, azim=-37.5)
        plt.show()

    @staticmethod
    def plot_comparison(P1, P2, fig, ms1=3, ms2=3):
        """
        Re-implementation of MATLAB’s plot_comparison.

        Plots P1 in blue and those points of P2 not in P1 in red.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # Compute set difference based on rows.
        # One way is to use a view with a structured dtype.
        def unique_rows(a):
            a = np.ascontiguousarray(a)
            b = a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
            _, idx = np.unique(b, return_index=True)
            return a[idx]

        dtype = np.dtype((np.void, P1.dtype.itemsize * P1.shape[1]))
        P1_view = P1.view(dtype).ravel()
        P2_view = P2.view(dtype).ravel()
        common = np.intersect1d(P1_view, P2_view)
        mask = np.array([row.view(dtype)[0] not in common for row in P2])
        P2_diff = P2[mask]

        plt.figure(fig)
        if P1.shape[1] == 3:
            ax = plt.axes(projection='3d')
            ax.plot(P1[:, 0], P1[:, 1], P1[:, 2], '.b', markersize=ms1)
            ax.plot(P2_diff[:, 0], P2_diff[:, 1], P2_diff[:, 2], '.r', markersize=ms2)
        else:
            plt.plot(P1[:, 0], P1[:, 1], '.b', markersize=ms1)
            plt.plot(P2_diff[:, 0], P2_diff[:, 1], '.r', markersize=ms2)
        plt.axis('equal')
        plt.show()

    @staticmethod
    def plot_branches(P, cover, segment, fig=1, ms=1, segind=1, BO=1000):
        """
        Re-implementation of MATLAB’s plot_branches.

        Plots the branch-segmented point cloud with each segment assigned a unique random color.

        Parameters
        ----------
        P : np.ndarray
            Point cloud.
        cover : dict‐like
            Must contain "ball" (list of cover-set indices).
        segment : dict‐like
            Must contain "segments", "ChildSegment", and "ParentSegment".
        segind : int, optional
            Starting segment index (1-indexed).
        BO : int, optional
            How many branching orders to plot (default 1000 = all).
        """
        import numpy as np
        import matplotlib.pyplot as plt

        Bal = cover["ball"]
        Segs = segment["segments"]
        SChi = segment["ChildSegment"]
        SPar = segment["ParentSegment"]

        # Flatten Segs if necessary.
        if isinstance(Segs[0], list):
            ns = len(Segs)
            Seg = []
            for s in Segs:
                flat = []
                for item in s:
                    flat.extend(item)
                Seg.append(np.array(flat, dtype=int))
        else:
            Seg = Segs
        ns = len(Seg)
        # Generate unique random colors.
        col = np.random.rand(ns, 3)
        for i in range(1, ns):
            parent_idx = int(SPar[i]) - 1
            c = col[i]
            while np.sum(np.abs(col[parent_idx] - c)) < 0.2:
                c = np.random.rand(3)
            col[i] = c

        segments = [segind]
        C = SChi[int(segind) - 1]
        b = 0
        while C and b <= BO:
            b += 1
            segments.extend(C)
            new_C = []
            for seg in C:
                new_C.extend(SChi[int(seg) - 1])
            C = new_C

        fig_obj = plt.figure(fig)
        ax = plt.axes(projection='3d')
        for seg in segments:
            idx = int(seg) - 1
            # Collect points from all cover sets referenced in this segment.
            pts = np.concatenate([np.array(Bal[int(j) - 1]) for j in np.array(Seg[idx])])
            ax.scatter(P[pts, 0], P[pts, 1], P[pts, 2], c=[col[idx]], marker='.', s=ms)
        ax.set_box_aspect((1, 1, 1))
        plt.show()

    @staticmethod
    def plot2d(X, Y, fig, strtit, strx, stry, leg=None, E=None):
        """
        Re-implementation of MATLAB’s plot2d.

        Plots 2D data with optional error bars.

        Parameters
        ----------
        X : list or array
            X data (if empty, indices are used).
        Y : list or array
            Y data. Can be a 2D array or a list of arrays.
        fig : int
            Figure number.
        strtit : str
            Title.
        strx : str
            X-axis label.
        stry : str
            Y-axis label.
        leg : list, optional
            Legend entries.
        E : list or array, optional
            Error bars (must match Y).
        """
        import numpy as np
        import matplotlib.pyplot as plt

        lw = 1.5
        plt.figure(fig)
        if E is None:
            # Without error bars.
            if X is not None and len(X) > 0:
                for i in range(len(Y)):
                    plt.plot(X[i], Y[i], linewidth=lw)
            else:
                for i in range(len(Y)):
                    plt.plot(Y[i], linewidth=lw)
        else:
            # With error bars.
            if X is not None and len(X) > 0:
                for i in range(len(Y)):
                    plt.errorbar(X[i], Y[i], yerr=E[i], linewidth=lw)
            else:
                for i in range(len(Y)):
                    plt.errorbar(np.arange(len(Y[i])), Y[i], yerr=E[i], linewidth=lw)
        plt.title(strtit, fontsize=12, fontweight='bold')
        plt.xlabel(strx, fontsize=12, fontweight='bold')
        plt.ylabel(stry, fontsize=12, fontweight='bold')
        if leg is not None:
            plt.legend(leg, loc='best')
        plt.grid(True)
        plt.show()


    @staticmethod
    def plot_cylinder_model2(cylinder, fig=1, nf=20, alp=1, Ind=None):
        """
        Re-implementation of MATLAB’s plot_cylinder_model2.

        Similar to plot_cylinder_model but handles a top radius (for truncated cones).

        Parameters
        ----------
        cylinder : dict‐like
            Must contain "radius", "TopRadius", "length", "start", "axis", "BranchOrder".
        fig : int, optional
            Figure number.
        nf : int, optional
            Number of facets.
        alp : float, optional
            Alpha value.
        Ind : array-like, optional
            Indices of cylinders to plot.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        Rad = np.array(cylinder["radius"])
        Rad2 = np.array(cylinder["TopRadius"])
        Len = np.array(cylinder["length"])
        Sta = np.array(cylinder["start"])
        Sta = Sta - Sta[0]
        Axe = np.array(cylinder["axis"])
        BOrd = np.array(cylinder["BranchOrder"])
        if Ind is not None:
            Rad = Rad[Ind]
            Len = Len[Ind]
            Sta = Sta[Ind, :]
            Axe = Axe[Ind, :]
            BOrd = BOrd[Ind]
        nc = Rad.shape[0]

        base_col = np.array([
            [0.00, 0.00, 1.00],
            [0.00, 0.50, 0.00],
            [1.00, 0.00, 0.00],
            [0.00, 0.75, 0.75],
            [0.75, 0.00, 0.75],
            [0.75, 0.75, 0.00],
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
            [0.95, 0.95, 0.00],
            [0.25, 0.25, 0.75],
            [0.75, 0.75, 0.75],
            [0.00, 1.00, 0.00],
            [0.76, 0.57, 0.17],
            [0.54, 0.63, 0.22],
            [0.34, 0.57, 0.92],
            [1.00, 0.10, 0.60],
            [0.88, 0.75, 0.73],
            [0.10, 0.49, 0.47],
            [0.66, 0.34, 0.65],
            [0.99, 0.41, 0.23]
        ])
        N = int(np.max(BOrd)) + 1
        if N <= 20:
            col = base_col[:N, :]
        else:
            m = int(np.ceil(N / 20))
            col = np.tile(base_col, (m, 1))[:N, :]

        Cir = {}
        for i in range(4, nf + 1):
            theta = np.linspace(0, 2 * np.pi, i, endpoint=False)
            B = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(i)))
            T = np.column_stack((np.cos(theta), np.sin(theta), np.ones(i)))
            Cir[i] = (np.vstack((B, T)), np.column_stack((
                np.arange(1, i + 1),
                np.arange(i + 1, 2 * i + 1),
                np.concatenate((np.arange(i + 2, 2 * i + 1), [i + 1])),
                np.concatenate((np.arange(2, i + 1), [1]))
            )))

        max_vertices = 2 * nc * (nf + 1)
        Vert = np.zeros((max_vertices, 3))
        max_facets = nc * (nf + 1)
        Facets = np.zeros((max_facets, 4), dtype=int)
        fvd = np.zeros((max_facets, 3))
        t = 0
        f = 0

        def rotation_matrix(axis, angle):
            axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) != 0 else axis
            a = np.cos(angle / 2)
            b, c, d = -axis * np.sin(angle / 2)
            return np.array([
                [a * a + b * b - c * c - d * d, 2 * (b * c - a * d),       2 * (b * d + a * c)],
                [2 * (b * c + a * d),       a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                [2 * (b * d - a * c),       2 * (c * d + a * b),       a * a + d * d - b * b - c * c]
            ])
        def mat_vec_subtraction(mat, vec):
            return mat - vec

        for i in range(nc):
            n_facets = int(np.ceil(np.sqrt(Rad[i] / Rad[0]) * nf))
            n_facets = min(n_facets, nf)
            n_facets = max(n_facets, 4)
            C = np.copy(Cir[n_facets][0])
            m_total = C.shape[0]
            half = m_total // 2
            C[:half, :2] = Rad[i] * C[:half, :2]
            C[half:, :2] = Rad2[i] * C[half:, :2]
            C[n_facets:, 2] = Len[i] * C[n_facets:, 2]
            ang = np.arccos(np.clip(Axe[i, 2], -1, 1))
            axis_rot = np.cross(np.array([0, 0, 1]), Axe[i])
            Rot = rotation_matrix(axis_rot, ang) if np.linalg.norm(axis_rot) >= 1e-6 else np.eye(3)
            C = (Rot @ C.T).T
            C = mat_vec_subtraction(C, -Sta[i])
            n_vert = m_total
            Vert[t:t + n_vert, :] = C
            Facets[f:f + n_facets, :] = Cir[n_facets][1] + t
            fvd[f:f + n_facets, :] = np.tile(col[int(BOrd[i]) + 1], (n_facets, 1))
            t += n_vert
            f += n_facets
        Vert = Vert[:t, :]
        Facets = Facets[:f, :]
        fvd = fvd[:f, :]

        fig_obj = plt.figure(fig)
        ax = plt.axes(projection='3d')
        ax.plot3D([Vert[0, 0]], [Vert[0, 1]], [Vert[0, 2]], 'o')
        poly = Poly3DCollection(Vert[Facets - 1], facecolors=fvd, edgecolor='none', alpha=alp)
        ax.add_collection3d(poly)
        ax.set_box_aspect((1, 1, 1))
        ax.grid(True)
        ax.view_init(elev=30, azim=-37.5)
        plt.show()

    @staticmethod
    def plot_distribution(QSM, fig, rela, cumu, dis, dis2=None, dis3=None, dis4=None):
        """
        Re-implementation of MATLAB’s plot_distribution.

        Plots one or more distributions (from QSM.treedata) as a bar plot.
        If rela==1, values are converted to percentages; if cumu==1, cumulative sums are plotted.

        Parameters
        ----------
        QSM : object or list
            If a list, each element is assumed to have a .treedata attribute.
            Otherwise, QSM.treedata must contain the field named by dis.
        fig : int
            Figure number.
        rela : int
            If 1, plot relative values (%).
        cumu : int
            If 1, plot cumulative distribution.
        dis : str
            Name of the first distribution (e.g. 'VolCylDia').
        dis2, dis3, dis4 : str, optional
            Additional distribution field names.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # Generate labels based on the distribution field names.
        if dis.startswith('Vol'):
            str_desc = 'volume'
            ylab = 'Volume (L)'
        elif dis.startswith('Are'):
            str_desc = 'area'
            ylab = 'Area (m^2)'
        elif dis.startswith('Len'):
            str_desc = 'length'
            ylab = 'Length (m)'
        elif dis.startswith('Num'):
            str_desc = 'number'
            ylab = 'Number'
        else:
            str_desc = ''
            ylab = ''
        if dis.endswith('Dia'):
            str2 = 'diameter'
            xlab = 'diameter (cm)'
        elif dis.endswith('Hei'):
            str2 = 'height'
            xlab = 'height (m)'
        elif dis.endswith('Ord'):
            str2 = 'order'
            xlab = 'order'
        elif dis.endswith('Ang'):
            str2 = 'angle'
            xlab = 'angle (deg)'
        elif dis.endswith('Azi'):
            str2 = 'azimuth direction'
            xlab = 'azimuth direction (deg)'
        elif dis.endswith('Zen'):
            str2 = 'zenith direction'
            xlab = 'zenith direction (deg)'
        else:
            str2 = ''
            xlab = ''

        # Collect distribution data.
        if isinstance(QSM, list):
            m = len(QSM)
            D = np.array(QSM[0].treedata[dis])
            n = D.shape[1]
            for i in range(1, m):
                d = np.array(QSM[i].treedata[dis])
                if d.shape[1] > n:
                    n = d.shape[1]
                    D = np.pad(D, ((0, 0), (0, n - D.shape[1])), 'constant')
                    D[i, :d.shape[1]] = d
                elif d.shape[1] < n:
                    d = np.pad(d, ((0, 0), (0, n - d.shape[1])), 'constant')
                    D[i, :] = d
                else:
                    D[i, :] = d
        else:
            m = 1
            D = np.array(QSM['treedata'][dis])
            n = D.shape[0]
            if D.size == 0 or np.all(D == 0):
                return
            if dis2 is not None:
                D2 = np.array(QSM['treedata'][dis2])
                if m < 2:
                    D = np.vstack((D, D2))
                elif dis3 is not None:
                    D3 = np.array(QSM['treedata'][dis3])
                    D = np.vstack((D, D2, D3))
                elif dis4 is not None:
                    D4 = np.array(QSM['treedata'][dis4])
                    D = np.vstack((D, D2, D3, D4))
        if rela:
            for i in range(D.shape[0]):
                total = np.sum(D[i, :])
                if total > 0:
                    D[i, :] = D[i, :] / total * 100
            ylab = 'Relative value (%)'
        if cumu:
            D = np.cumsum(D, axis=1)

        plt.figure(fig)
        if dis.endswith('Azi') or dis.endswith('hAzi') or dis.endswith('1Azi'):
            x = np.arange(-170, 181, 10)
        elif dis.endswith('Zen') or dis.endswith('Ang'):
            x = np.arange(10, 10 * n + 1, 10)
        else:
            x = np.arange(1, n + 1)
        # d =D.T

        if len(D.shape) >1:
            rang= max(np.max(x)-np.min(x),100)
            for i,row in enumerate(D):
                plt.bar(x, row,width=1*rang//100)
        else:
            rang= max(np.max(x)-np.min(x),100)
            for i,row in enumerate(D):
                
                plt.bar(x[i], row,width=1*rang//100)
        if dis.endswith('Cyl'):
            xlab = 'Cylinder ' + xlab
        else:
            xlab = 'Branch ' + xlab
        plt.title('Tree segment ' + str_desc + ' per ' + str2 + ' class')
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.axis('tight')
        plt.grid(True)
        if m > 1:
            L = ['model' + str(i + 1) for i in range(m)]
            plt.legend(L, loc='best')
        # plt.show()


    @staticmethod
    def plot_large_point_cloud(P, fig, ms, rel):
        """
        Re-implementation of MATLAB’s plot_large_point_cloud.

        Plots a random subset of a large point cloud. The input "rel" is given in
        percentage points (e.g. if rel==12, roughly 12% of points are plotted).

        Parameters
        ----------
        P : np.ndarray
            Point cloud.
        fig : int
            Figure number.
        ms : int
            Marker size.
        rel : float
            Subset size in percentage.
        """
        # Compute coefficient (following MATLAB’s logic).
        coeff = 0.5 / (1 - rel / 100)
        I = np.random.rand(P.shape[0]) < coeff
        # Assuming you already have a method plot_point_cloud available:
        PlottingUtils.plot_point_cloud(P[I, :], fig=fig, ms=ms)