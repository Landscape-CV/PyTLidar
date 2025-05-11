import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

"""
Python adaptation and extension of TREEQSM:

Calculates the distances of points to a line in 3D space.


% -----------------------------------------------------------
% This file is part of TREEQSM.
%
% TREEQSM is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% TREEQSM is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY without even the implied warranty of
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
def growth_volume_correction(cylinder, inputs):
    print('----------')
    print('Growth volume based correction of cylinder radii:')
    
    Rad = np.array(cylinder['radius'], dtype=float)
    Rad0 = Rad.copy()
    Len = np.array(cylinder['length'], dtype=float)
    CPar = np.array(cylinder['parent'], dtype=int)
    CExt = np.array(cylinder['extension'], dtype=int)
    
    initial_volume = round(1000 * np.pi * np.sum(Rad**2 * Len))
    print(f' Initial_volume (L): {initial_volume}')
    
    # Define the child cylinders for each cylinder
    n = len(Rad)
    CChi = [[] for _ in range(n)]
    for i in range(n):
        CChi[i] = np.where(CPar == i)[0]
    
    # Compute the growth volume
    GrowthVol = np.zeros(n)
    S = [len(CChi[i]) for i in range(n)]
    modify = np.array(S) == 0
    GrowthVol[modify] = np.pi * Rad[modify]**2 * Len[modify]
    
    parents = np.unique(CPar[modify])
    if parents[0] == 0:
        parents = parents[1:]
    
    while len(parents) > 0:
        V = np.pi * Rad[parents]**2 * Len[parents]
        for i, parent in enumerate(parents):
            GrowthVol[parent] = V[i] + np.sum(GrowthVol[CChi[parent]])
        parents = np.unique(CPar[parents])
        if parents[0] == 0:
            parents = parents[1:]
    
    # Fit the allometry: Rad = a * GrowthVolume^b + c
    def allometry(x, xdata):
        return x[0] * xdata**x[1] + x[2]

    # Fit model parameters
    popt, _ = curve_fit(allometry, GrowthVol, Rad, p0=[0.5, 0.5, 0])
    
    print(' Allometry model parameters R = a * GV^b + c:')
    print(f'   Multiplier a: {popt[0]}')
    print(f'   Exponent b: {popt[1]}')
    if len(popt) > 2:
        print(f'   Intersect c: {popt[2]}')
    
    # Compute the predicted radius from the allometry
    PredRad = allometry(popt, GrowthVol)
    
    # Correct the radii based on the predictions
    fac = inputs['GrowthVolFac']
    modify = (Rad < PredRad / fac) | (Rad > fac * PredRad)
    modify[(Rad < PredRad / fac) & (CExt == 0)] = False  # Do not increase the radius at tips
    CorRad = PredRad[modify]
    
    # Plot allometry and radii modification
    gvm = np.max(GrowthVol)
    gv = np.linspace(0, gvm, 1000)
    PRad = allometry(popt, gv)
    
    plt.figure(1)
    plt.plot(GrowthVol, Rad, '.b', markersize=2)
    plt.plot(gv, PRad, '-r', linewidth=2)
    plt.plot(gv, PRad / fac, '-g', linewidth=2)
    plt.plot(gv, fac * PRad, '-g', linewidth=2)
    plt.grid(True)
    plt.xlabel('Growth volume (m^3)')
    plt.ylabel('Radius (m)')
    plt.legend(['radius', 'predicted radius', 'minimum radius', 'maximum radius'], loc='NorthWest')

    plt.figure(2)
    plt.hist(CorRad - Rad[modify])
    plt.xlabel('Change in radius')
    plt.title('Number of cylinders per change in radius class')
    plt.show()
    
    # Determine the maximum radius change
    R = Rad[modify]
    D = np.max(np.abs(R - CorRad))  # Maximum radius change
    J = np.abs(R - CorRad) == D
    D = CorRad[J] - R[J]
    
    # Modify the radius according to allometry
    Rad[modify] = CorRad
    cylinder['radius'] = Rad
    
    print(f' Modified {np.sum(modify)} of the {n} cylinders')
    print(f' Largest radius change (cm): {round(1000 * D) / 10}')
    corrected_volume = round(1000 * np.pi * np.sum(Rad**2 * Len))
    print(f' Corrected volume (L): {corrected_volume}')
    print(f' Change in volume (L): {corrected_volume - initial_volume}')
    print('----------')

    return cylinder