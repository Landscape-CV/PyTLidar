"""
Python adaptation and extension of TREEQSM:

Determine connected components from cover sets using neighbor relations.


% This file is part of TREEQSM.
%
% TREEQSM is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% TREEQSM is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with TREEQSM.  If not, see <http://www.gnu.org/licenses/>.

Version: 0.0.1
Date: 9 Feb 2025
Authors: Fan Yang, John Hagood, Amir Hossein Alikhah Mishamandani
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""

from collections import deque
from typing import List, Union, Tuple
import numpy as np

def connected_components(
        Nei: List[List[int]],
        Sub: Union[List[int], List[bool], int] = 0,
        MinSize: int = 1,
) -> Tuple[List[List[int]], List[int]]:
    """
    Determine connected components from cover sets using neighbor relations.

    Parameters:
        Nei (List[List[int]]):
            A list with length n_sets, where each entry is a list of neighbor indices.
        Sub (Union[List[int], List[bool], int]):
            Defines the subset of cover sets to examine. It can be:
              - A list of indices (int) to include.
              - A boolean list of length n_sets, with True marking sets to include.
              - An integer 0 to indicate "all cover sets".
        MinSize (int):
            Minimum number of cover sets required for a connected component.

    Returns:
        Components (List[List[int]]):
            A list of connected components. Each component is a list of cover set indices.
        CompSize (List[int]):
            A list with the sizes of each connected component.
    """

    n_sets = len(Nei)

    # Process 'Sub' into a boolean mask of length n_sets.
    if isinstance(Sub, int):
        if Sub == 0:
            sub_mask = [True] * n_sets
        else:
            # If Sub is an integer > 0, treat it as a singleton (1-element list)
            sub_mask = [False] * n_sets
            if 0 <= Sub < n_sets:
                sub_mask[Sub] = True
            else:
                raise ValueError("Sub index out of range.")
    elif isinstance(Sub, list):
        if all(isinstance(s, bool) for s in Sub):
            if len(Sub) != n_sets:
                raise ValueError("Length of boolean Sub does not match number of cover sets.")
            sub_mask = Sub.copy()
        else:
            # Assume list of indices (convert to boolean mask)
            sub_mask = [False] * n_sets
            for idx in Sub:
                if 0 <= idx < n_sets:
                    sub_mask[idx] = True
                else:
                    raise ValueError("An index in Sub is out of range.")
    else:
        raise TypeError("Sub must be a list of indices, a list of booleans, or 0 for all sets.")

    # Special handling for very small subsets (length <= 3)
    # Convert sub_mask to a list of indices for the subset.
    indices = [i for i, flag in enumerate(sub_mask) if flag]

    if len(indices) <= 3:
        # Handle 1, 2, or 3 cover sets.
        components = []
        comp_sizes = []
        if len(indices) == 1:
            components.append([indices[0]])
            comp_sizes.append(1)
        elif len(indices) == 2:
            i, j = indices
            # Check if they are connected
            if j in Nei[i]:
                components.append([i, j])
                comp_sizes.append(2)
            else:
                components.append([i])
                components.append([j])
                comp_sizes.extend([1, 1])
        else:  # len(indices) == 3
            i, j, k = indices
            # Check connections among three nodes
            conn_ij = j in Nei[i]
            conn_ik = k in Nei[i]
            conn_jk = k in Nei[j]
            if sum([conn_ij, conn_ik, conn_jk]) >= 2:
                components.append([i, j, k])
                comp_sizes.append(3)
            elif conn_ij:
                components.append([i, j])
                components.append([k])
                comp_sizes.extend([2, 1])
            elif conn_ik:
                components.append([i, k])
                components.append([j])
                comp_sizes.extend([2, 1])
            elif conn_jk:
                components.append([j, k])
                components.append([i])
                comp_sizes.extend([2, 1])
            else:
                components.append([i])
                components.append([j])
                components.append([k])
                comp_sizes.extend([1, 1, 1])
        # Filter by minimum size:
        filtered_components = []
        filtered_sizes = []
        for comp, size in zip(components, comp_sizes):
            if size >= MinSize:
                filtered_components.append(comp)
                filtered_sizes.append(size)
        return filtered_components, filtered_sizes

    # For the general case, we use a BFS strategy.
    Components = []
    CompSize = []

    # A set to easily check membership; tracks which nodes are still unvisited.
    unvisited = {i for i, flag in enumerate(sub_mask) if flag}

    while unvisited:
        # Start a new component from an arbitrary unvisited node.
        start = unvisited.pop()
        comp = [start]
        # Queue for BFS.
        queue = deque([start])
        while queue:
            current = queue.popleft()
            # For each neighbor of the current node:
            for neighbor in Nei[current]:
                # Process neighbor if it is in the subset and unvisited.
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    comp.append(neighbor)
                    queue.append(neighbor)
        # Record component if it meets the minimum size.
        if len(comp) >= MinSize:
            Components.append(comp)
            CompSize.append(len(comp))

    return Components, CompSize

def connected_components_array(Nei, Sub, MinSize, Fal=None):
    """
    Version of connected components using Numpy and accepting Fal as an input

    Inputs:
    Nei       : List of neighboring cover sets for each cover set (list of lists or list of arrays)
    Sub       : Subset whose components are determined. 
                If length(Sub) <= 3 and not a logical array, it is treated as a small subset.
                If Sub is a single 0, it means all cover sets.
                Otherwise, Sub is a logical array or a list of indices.
    MinSize   : Minimum number of cover sets in an acceptable component.
    Fal       : Logical false vector for the cover sets (optional).

    Outputs:
    Components: List of connected components (list of arrays).
    CompSize  : Number of sets in the components (list of integers).
    """
    Sub = Sub.copy()
    if len(Sub) <= 3 and not isinstance(Sub, (np.ndarray, list)) and Sub[0] > 0:
        # Very small subset, i.e., at most 3 cover sets
        n = len(Sub)
        if n == 1:
            Components = [np.array(Sub, dtype=np.uint32)]
            CompSize = [1]
        elif n == 2:
            if Sub[1] in Nei[Sub[0]]:
                Components = [np.array(Sub, dtype=np.uint32)]
                CompSize = [1]
            else:
                Components = [np.array([Sub[0]], dtype=np.uint32), np.array([Sub[1]], dtype=np.uint32)]
                CompSize = [1, 1]
        elif n == 3:
            I = Sub[1] in Nei[Sub[0]]
            J = Sub[2] in Nei[Sub[0]]
            K = Sub[2] in Nei[Sub[1]]
            if I + J + K >= 2:
                Components = [np.array(Sub, dtype=np.uint32)]
                CompSize = [1]
            elif I:
                Components = [np.array([Sub[0], Sub[1]], dtype=np.uint32), np.array([Sub[2]], dtype=np.uint32)]
                CompSize = [2, 1]
            elif J:
                Components = [np.array([Sub[0], Sub[2]], dtype=np.uint32), np.array([Sub[1]], dtype=np.uint32)]
                CompSize = [2, 1]
            elif K:
                Components = [np.array([Sub[1], Sub[2]], dtype=np.uint32), np.array([Sub[0]], dtype=np.uint32)]
                CompSize = [2, 1]
            else:
                Components = [np.array([Sub[0]], dtype=np.uint32), np.array([Sub[1]], dtype=np.uint32), np.array([Sub[2]], dtype=np.uint32)]
                CompSize = [1, 1, 1]
        return Components, CompSize

    elif any(Sub) or (len(Sub) == 1 and Sub[0] == 0):
        nb = len(Nei)
        if Fal is None:
            Fal = np.zeros(nb, dtype=bool)
        Fal = Fal.copy()
        if len(Sub) == 1 and Sub[0] == 0:
            # All the cover sets
            ns = nb
            Sub = ~Fal
        elif not isinstance(Sub, (np.ndarray, list)):
            # Subset of cover sets
            ns = len(Sub)
            sub = np.zeros(nb, dtype=bool)
            sub[Sub] = True
            Sub = sub
        else:
            # Subset of cover sets
            ns = np.sum(Sub)

        Components = []
        CompSize = []
        nc = 0  # number of components found
        m = 0
        while m < nb and not Sub[m]:
            m += 1
        i = 0
        Comp = np.zeros(ns, dtype=np.uint32)
        while i < ns:
            Add = Nei[m]
            I = Sub[Add]
            Add = Add[I]
            a = len(Add)
            Comp = Comp.copy()
            Comp[0] = m
            Sub[m] = False
            t = 1
            while a > 0:
                if t+a > len(Comp):
                    Comp = np.concatenate([Comp,np.zeros((t+a-len(Comp)))])
                Comp[t:t + a] = Add
                Sub[Add] = False
                t += a
                Add = np.concatenate([Nei[a] for a in Add])
                I = Sub[Add]
                Add = Add[I]
                # Select the unique elements of Add
                # n = len(Add)
                # if n > 2:
                #     I = np.ones(n, dtype=bool)
                #     for j in range(n):
                #         if not Fal[Add[j]]:
                #             Fal[Add[j]] = True
                #         else:
                #             I[j] = False
                #     Fal[Add] = False
                #     Add = Add[I]
                # elif n == 2:
                #     if Add[0] == Add[1]:
                #         Add = Add[:1]
                Add = np.unique(Add)
                a = len(Add)
            i += t
            if t >= MinSize:
                nc += 1
                Components.append(Comp[:t])
                CompSize.append(t)
            if i < ns:
                while m < nb and not Sub[m]:
                    m += 1
        return Components, np.array(CompSize)
    else:
        return [], 0