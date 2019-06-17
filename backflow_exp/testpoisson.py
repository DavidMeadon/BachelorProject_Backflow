from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import scipy as sp
import scipy.sparse.linalg as ssl
    # Assignment: SUPG

def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def SolAdvDif(mu,N,nn,boundy):

    mesh_root = 'stenosis_f0.6'
    mesh_root += '_fine'
    mesh = Mesh(mesh_root + '.xml')
    tol 		= 	1E-14
    #h			=	mesh.hmin()
    h			=	1./N
    b			= 	Constant([1/np.sqrt(2),1/np.sqrt(2)])
    #f			= 	Expression('x[0]<=0.1 && x[1]<=0.1 ? 10 : 0', degree=2)
    f           =   Constant(1)
    u_D 		= 	Constant(0.0)
    V 			= 	FunctionSpace(mesh, 'P', nn)
    u			=	TrialFunction(V)
    v			= 	TestFunction(V)

    a 			= 	mu*inner(grad(u), grad(v))*dx
    L 			= 	f*v*dx


    u			=	Function(V)


    def boundary(x, on_boundary):
        return on_boundary

    
    bcs 			= 	DirichletBC(V, u_D, boundary)


    lap = assemble(a)
    bcs.apply(lap)
    lapmat = np.array(lap.array())
    print(is_pos_def(lapmat))
    # lapmat = np.array(lap.array())
    lapmat2 = sp.sparse.bsr_matrix(lap.array())
    print(lapmat2.shape)
    smalleig = ssl.eigs(lapmat2, 6, which='SM', return_eigenvectors=False)
    print(smalleig)
    fig = plt.figure()
    axes = plt.gca()
    axes.set_xlim([-0.0001, 0.0005])
    axes.set_ylim([-1, 1])
    for EV in smalleig:
        plt.plot(EV.real, EV.imag, 'go')
    plt.show()
    # eigenvals = LA.eigvals(laplace_backflow_final)





    solve(a == L, u, bcs)








    return [u,mesh]

def plotSolution(u,mu,mesh,boundy):
    c			=	plot(u)
    cca = 'Dirichlet'
    if boundy==2:
        cca='Neumann'


    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.colorbar(c)
    plt.show()




N				=	32*1 + 118*0 + 88*0 + 230*0		# Number of nodes = NxN
nn				=	1							# Degree of Finite Elements
case			=	2							# 1: WHITOUT SUPG   2: WITH SUPG
boundy			=	1							# 1: DIRICHLET		2: NEUMANN
mu				= 	Constant(1e-3)				# Viscosity


[u , mesh ]	=	SolAdvDif(mu,N,nn,boundy) 

plotSolution(u,mu,mesh,boundy)


    # Pe = 2.3   h in the direction /   1.5   h = 1/N
    # Pe = 5.6   h in the direction /   3.0   h = 1/N





