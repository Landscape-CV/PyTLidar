"""
Python adaptation and extension of TREEQSM:

Display the two times given. T1 is the time named with the "string" and
    T2 is named "Total".


% -----------------------------------------------------------
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
% -----------------------------------------------------------


Version: 0.0.1
Date: 18 Jan 2025
Authors: Fan Yang, John Hagood, Amir Hossein Alikhah Mishamandani
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""


from tools.sec2min import sec2min
import sys

def display_time(T1, T2, string, display):
    """
    Display the two times given. T1 is the time named with the "string" and
    T2 is named "Total".
    """
    if display:
        tmin, tsec = sec2min(T1)
        Tmin, Tsec = sec2min(T2)

        if tmin < 60 and Tmin < 60:
            if tmin < 1 and Tmin < 1:
                result = f"{string} {tsec} sec.   Total: {Tsec} sec"
            elif tmin < 1:
                result = f"{string} {tsec} sec.   Total: {Tmin} min {Tsec} sec"
            else:
                result = f"{string} {tmin} min {tsec} sec.   Total: {Tmin} min {Tsec} sec"
        elif tmin < 60:
            Thour = Tmin // 60
            Tmin %= 60
            result = f"{string} {tmin} min {tsec} sec.   Total: {Thour} hours {Tmin} min"
        else:
            thour = tmin // 60
            tmin %= 60
            Thour = Tmin // 60
            Tmin %= 60
            result = f"{string} {thour} hours {tmin} min.   Total: {Thour} hours {Tmin} min"

        sys.stdout.write(result+'\n')