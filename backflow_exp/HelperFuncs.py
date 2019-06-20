from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.sparse.linalg as ssl
from numpy import linalg as LA
from math import fabs

def abs_n(x):
    return 0.5 * (x - abs(x))


# # TODO rework the backflowarea function since it doesn't do as it should.
# def backflowarea(a, b, c):
#     if assemble(abs_n(dot(a, b)) * c) > 0:
#         return 1
#     else:
#         return 0


def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            LA.cholesky(A)
            print("Positive Definite")
            return True
        except LA.LinAlgError:
            print("Not Positive Definite")
            return False
    else:
        print('Not Symmetric')
        return False


def diag_dom(X):
    D = np.diag(np.abs(X)) # Find diagonal coefficients
    S = np.sum(np.abs(X), axis=1) - D # Find row sum without diagonal
    if np.all(D >= S):
        sol = 'Matrix is diagonally dominant'
        return sol
        print(sol)
    else:
        sol = 'NOT diagonally dominant'
        return sol
        print(sol)
    return


# def test(a, b):
#     print(dot(a, b))
#     return 1


# def betaupdate(current_sol, prev_sol, facet_norm, dx, beta):
#     if assemble(abs_n(dot(current_sol, facet_norm)) * div(current_sol) * dx) > assemble(abs_n(dot(prev_sol, facet_norm)) * div(prev_sol) * dx):
#         return min(max(assemble(beta * dx) + 0.1, 0.2), 1)
#     else:
#         return min(max(assemble(beta * dx) - 0.1, 0.2), 1)


# def radfinder(M, k, n):
#     tote = 0
#     for p in range(n):
#         if p != k:
#             tote += np.abs(M[p])
#     return tote
#
#
# def gershgorinplotter(mat, t, bfsm):
#     n = mat.shape[1]
#     # radii = np.zeros((n, 1))
#     # centers = np.zeros((n,1))
#     fig = plt.figure()
#     axes = plt.gca()
#     axes.set_xlim([-10, 10])
#     axes.set_ylim([-10, 10])
#     if n > 0:
#         for k in range(n):
#             # centers[k] = mat[k, k]
#             center = mat[k, k]
#             rad = radfinder(mat[k, :], k, n)
#             # radii[k] = radfinder(mat[k, :], k, n)
#             circle1 = plt.Circle((center, 0), rad)
#             plt.gcf().gca().add_artist(circle1)
#     plt.title('Method: ' + bfsm + ', Time: ' + str(t))
#     return fig

def isSquare(m):
    cols = len(m)
    for row in m:
        if len(row) != cols:
            return False
    return True

def GregsCircles(matrix):
    if isSquare(matrix) != True:
        print('Your input matrix is not square!')
        return []
    circles = []
    for x in range(0,len(matrix)):
        radius = 0
        piv = matrix[x][x]
        for y in range(0,len(matrix)):
            if x != y:
                radius += fabs(matrix[x][y])
        circles.append([piv,radius])
    return circles

def plotCircles(circles, t, bfsm, param, paramval):
    fig, ax = plt.subplots()
    plt.title('Method: ' + bfsm + ', Time: ' + str(t))
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    if circles == []:
        return fig
    index, radi = zip(*circles)
    Xupper = 10#max(index) + np.std(index)
    Xlower = -10#min(index) - np.std(index)
    Ylimit = max(radi) + np.std(index)
    ax = plt.gca()
    ax.cla()
    ax.set_xlim((Xlower,Xupper))
    ax.set_ylim((-Ylimit,Ylimit))
    plt.title('Method: ' + bfsm + ', Time: ' + str(t) + ', ' + param + ' = ' + str(paramval))
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    for x in range(0,len(circles)):
        circ = plt.Circle((index[x],0), radius = radi[x])
        ax.add_artist(circ)
    ax.plot([Xlower,Xupper],[0,0],'k--')
    ax.plot([0,0],[-Ylimit,Ylimit],'k--')
    return fig


def reduced_back_mat(backflow_func, lapmat):
    backflow_mat = assemble(lhs(backflow_func))
    backflow_vec = np.array(backflow_mat.array())
    reduced_laplace = lapmat * (backflow_vec != 0)
    laplace_backflow = reduced_laplace[~(reduced_laplace == 0).all(1)]
    return np.transpose(
        laplace_backflow.transpose()[~(laplace_backflow.transpose() == 0).all(1)])