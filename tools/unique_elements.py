"""
Python adaptation and extension of TREEQSM:

Return a list of unique elements from seq, preserving the original order.

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
import numpy as np
def unique_elements(seq):
    """
    Return a list of unique elements from seq, preserving the original order.

    Args:
        seq (iterable): An iterable of hashable elements.

    Returns:
        list: A list containing the unique elements in the order they first appear.
    """


    seen = set()
    unique = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique
def unique_elements_array(arr,False_mask=None):


    return np.unique(arr)


    if False_mask is None:
        False_mask = np.zeros((len(arr),)).astype(bool)
    False_mask = False_mask.copy()
    n = len(arr)
    
    if n > 2:
        I = [True] * n
        for j in range(n):
            if not False_mask[arr[j].astype(int)]:
                False_mask[arr[j].astype(int)] = True
            else:
                I[j] = False
        arr = [arr[i] for i in range(n) if I[i]]
        
    elif n == 2:
        if arr[0] == arr[1]:
            arr = [arr[0]]
    
    return np.array(arr)

# test
if __name__ == "__main__":
    data = [3, 5, 3, 's', 5, 6, 's']
    print(unique_elements(data))