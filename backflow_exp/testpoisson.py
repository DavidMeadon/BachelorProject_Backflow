from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
    # Assignment: SUPG

def SolAdvDif(mu,N,nn,boundy):

    mesh 		= 	UnitSquareMesh(N, N)
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
    lap2 = assemble(a)
    bcs.apply(lap)
    print(LA.norm((np.array(lap.array()) - np.array(lap2.array()))))





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





