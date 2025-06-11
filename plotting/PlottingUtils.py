"""
Python adaptation and extension of TREEQSM.

Version: 0.0.4
Date: 19 March 2025
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



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
    plt.title('Seg ' + str_desc + ' per\n' + str2 + ' class')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.axis('tight')
    plt.grid(True)
    if m > 1:
        L = ['model' + str(i + 1) for i in range(m)]
        plt.legend(L, loc='best')
    # plt.show()


def plot_branch_hist(QSM,fig,metric,bins = 20):
    """
        Plots histogram of branches according to chosen metric.

        Possible metrics include "order","parent","diameter","volume","area","length","angle","height","azimuth", and "zenith"
    """
    data = QSM["branch"][metric]
    data = data[data<.1]
    datahist = np.histogram(data,bins=bins)
    print(datahist)
    plt.figure(fig)
    # plt.stairs(datahist[0],datahist[1])
    plt.hist(data*100,bins=bins)
    plt.title(f"Quantity of Branches by {metric}")
    plt.xlabel("Branch Diameter (cm)")
    plt.ylabel("Quantity of Branches")


