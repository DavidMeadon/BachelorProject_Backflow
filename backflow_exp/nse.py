from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.sparse.linalg as ssl
from numpy import linalg as LA
import progress as prog
import HelperFuncs as HF


def nse(Re=1000, temam=False, bfs=False, level=1, velocity_degree=2, eps=0.0002, dt=0.001, plotcircles=0, gammagiv=1, betagiv=1):

    mesh_root = 'stenosis_f0.6'
    if level == 2:
        mesh_root += '_fine'

    mesh = Mesh(mesh_root + '.xml')
    boundaries = MeshFunction('size_t', mesh, mesh_root + '_facet_region.xml')
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    stabmethod = 'No stabilization'

    VE = VectorElement('P', mesh.ufl_cell(), velocity_degree)
    PE = FiniteElement('P', mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, VE * PE)

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    w = Function(W)
    u0, p0 = w.split()


    ####

    theta = Constant(1)
    k = Constant(1 / dt)
    mu = Constant(0.035)
    rho = Constant(1.2)
    eps = Constant(eps)





    U = (3 / 2) * 0.5 * Re * float(mu) / float(rho)

    print('\n Re = {}, U = {}\n'.format(Re, U))

    u_ = theta * u + (1 - theta) * u0
    theta_p = theta
    p_ = theta_p * p + (1 - theta_p) * p0

    laplace = mu * inner(grad(u_), grad(v)) * dx #Defined separately so it may be used later

    F = (
            k * rho * dot(u - u0, v) * dx
            + laplace
            - p_ * div(v) * dx + q * div(u) * dx
            + rho * dot(grad(u_) * u0, v) * dx
    )

    n = FacetNormal(mesh)
    h = CellDiameter(mesh)

    # eps = Constant(dt * float(mu) / (0.1 ** 2 * float(rho)))

    if temam:
        F += 0.5 * rho * div(u0) * dot(u_, v) * dx

    beta = Constant(betagiv)    #Defines the initial beta from the given value
    gamma = Constant(gammagiv) #Defines the initial gamma from the given value

    backflow_func = 0.5 * rho * HF.abs_n(dot(u0, n)) * dot(u_, v) * ds(2) 

    param = 'No param'

    G = 0 #This variable will contain the backflow stabilisation term
    ### Added here beta parameter to the stabilization terms which we can control
    if bfs == 1:
        param = 'beta' #Setting the string for the parameter being changed
        stabmethod = 'velocity-penalization'  #Setting the string for the stabilisation method being changed
        G = 0.5 * rho * beta * dot(u0, n) * dot(u_, v) * ds(2)
        F -= G
    elif bfs == 2:
        param = 'beta'
        stabmethod = 'velocity-penalization negative part'
        G = 0.5 * rho * beta * HF.abs_n(dot(u0, n)) * dot(u_, v) * ds(2)
        F -= G
    elif bfs == 3:
        param = 'gamma'
        stabmethod = 'tangential penalization'
        Ctgt = h ** 2
        G = gamma * Ctgt * 0.5 * rho * HF.abs_n(dot(u0, n)) * (
                Dx(u[0], 1) * Dx(v[0], 1) + Dx(u[1], 1) * Dx(v[1], 1)) * ds(2)
    elif bfs == 4:
        param = 'gamma'
        stabmethod = 'tangential penalization - 2016 paper'
        Ctgt = h ** 2

        maxiv = HF.maxUneg(W, mesh, u0) #Finding the max|u dot n|_ 
        maxi = Constant(maxiv)

        G = -1 * gamma * maxi * 0.5 * rho * Ctgt * (Dx(u[0], 1) * Dx(v[0], 1) + Dx(u[1], 1) * Dx(v[1], 1)) * ds(2)
        F -= G
    elif velocity_degree == 1 and float(eps):
        F += eps / mu * h ** 2 * inner(grad(p_), grad(q)) * dx

    stabilisationTest = backflow_func - G #This term will be the one worked with during the automation, it could have added terms
                                          #Such as the laplace or other stabilising terms.

    a = lhs(F)
    L = rhs(F)

    inflow = Expression(('sin(a*t*DOLFIN_PI)*U*(1 - pow(x[1], 2))', '0.'),
                        U=U, t=0., a=2.5, degree=2)
    # inflow = Expression(('U*(1 - pow(x[1], 2))', '0.'),
    #                     U=U, t=0., degree=2)
    bcs = [
        DirichletBC(W.sub(0), Constant((0, 0)), boundaries, 4),
        DirichletBC(W.sub(0), inflow, boundaries, 1),
        DirichletBC(W.sub(0).sub(1), Constant(0), boundaries, 3),
    ]

    A = assemble(a)

    #Below is needed for good plots
    if auto:
        runtype = "auto"
    elif param == 'beta':
        runtype = round(assemble(beta * ds(2)), 3)
    elif param == 'gamma':
        runtype = round(assemble(gamma * ds(2)))
    elif param == 'No param':
        runtype = "No param"

    #The below is for formatting that names of the output files to be used in paraview
    suf = 'bfs{}_tem{}_Re{}_'.format(int(bfs), int(temam), Re) + param + '_' + '{}_'.format(runtype)
    if velocity_degree == 1:
        suf = 'p1_' + suf
    suf = 'l{}_'.format(level) + suf

    xdmf_u = XDMFFile('results/u_' + suf + '.xdmf')
    xdmf_p = XDMFFile('results/p_' + suf + '.xdmf')
    xdmf_tau = XDMFFile('tau_sd_' + suf + '.xdmf')
    # xdmf_u.parameters['rewrite_function_mesh'] = False
    # xdmf_p.parameters['rewrite_function_mesh'] = False
    # xdmf_tau.parameters['rewrite_function_mesh'] = False

    u0.rename('u', 'u')
    p0.rename('p', 'p')

    # FINAL TIME
    T = 0.4

    pt = prog.progress_timer(description='Time Iterations', n_iter=40) #Used to have a progress bar
    # MAIN SOLVING LOOP
    eigvec = []#Used when finding eigenvalues with varying values of the parameters, i.e. when needing to run multiple times

    for t in np.arange(dt, T + dt, dt):
        
        ## Using Fenics to find the current solution
        inflow.t = t
        assemble(a, tensor=A)
        b = assemble(L)
        [bc.apply(A, b) for bc in bcs]
        solve(A, w.vector(), b)

        ## Creating all the plots
        if plotcircles > 0:
            if bfs == 4: # Update the max|u dot n|_
                maxiv = HF.maxUneg(W, mesh, u0)
                maxi.assign(maxiv)

            ## Creating matrix with just backflow term
            backflow_mat = assemble(lhs(backflow_func))
            backflow_vec = np.array(backflow_mat.array())

            ## Create backflow with stabilisation matrix
            stabTensor = assemble(lhs(stabilisationTest))
            # for bc in bcs: bc.apply(stabTensor) %% If including the Laplace term then need to apply BC's
            stabMatrix = np.array(stabTensor.array())

            ## Creating reduced Stabilisation Matrix
            if backflow_vec.any():
                maxidx2 = np.array([np.nonzero(backflow_vec)[0].max(), np.nonzero(backflow_vec)[1].max()]).max()
                minidx2 = np.array([np.nonzero(backflow_vec)[0].min(), np.nonzero(backflow_vec)[1].min()]).min()
                reduced_stabMatrix = stabMatrix[minidx2:maxidx2 + 1, minidx2:maxidx2 + 1]
            else:
                reduced_stabMatrix = np.resize(np.array([]), [1, 1])

            ## Finding eigenvalues of reduced matrix and making the Gershgorin circles
            eigenvals_reduced_stabMatrix = LA.eigvals(reduced_stabMatrix)
            circles = HF.GregsCircles(reduced_stabMatrix)
            fig = HF.plotCircles(circles, round(t, 2), stabmethod, param, runtype)

            ## If want the minimum eigenvalues of the entire matrix and plot them
            if plotcircles == 2 and eigenvals_reduced_stabMatrix.size > 0 and stabMatrix.any():
                stabMatrix_sparse = sp.sparse.bsr_matrix(stabMatrix)  # Sparse Version
                small_eigenvals_stabMatrix = ssl.eigs(stabMatrix_sparse, 5, sigma=-10, which='LM', return_eigenvectors=False, v0=np.ones(stabMatrix_sparse.shape[0]))
                printlab = True
                for EV in small_eigenvals_stabMatrix:
                    if printlab:
                        plt.plot(EV.real, EV.imag, 'ko', label='Eigenvalues of full Matrix')
                        printlab = False
                    else:
                        plt.plot(EV.real, EV.imag, 'ko', label='_nolegend_')
                del stabMatrix_sparse, small_eigenvals_stabMatrix
            
            ## Plotting the eigenvalues of the reduced matrix
            printlab = True
            for eigval in eigenvals_reduced_stabMatrix:
                if printlab:
                    plt.plot(eigval.real, eigval.imag, 'r+', label='Eigenvalues of reduced Matrix')
                    printlab = False
                else:
                    plt.plot(eigval.real, eigval.imag, 'r+', label='_nolegend_')
            if eigenvals_reduced_stabMatrix.size > 0:
                plt.legend()
            fig.savefig('circles/' + str(round(t * 100)) + 'gersh.png')
            del stabTensor, stabMatrix, reduced_stabMatrix, eigenvals_reduced_stabMatrix, backflow_mat, backflow_vec
            plt.close(fig)


        xdmf_u.write(u0, t)
        xdmf_p.write(p0, t)
        pt.update()
    pt.finish()
    return eigvec

    del xdmf_u, xdmf_p


if __name__ == '__main__':
    Re = [5000]
    betaVec = [1] #Here an array of different beta values can be specified
    # gammaVec = [1] %If want to use a method relying on gamma
    eigsarr = []

    for Re_ in Re:
        for beta in betaVec:
            eigsarr.append(nse(Re_, level=1, temam=True, bfs=0, velocity_degree=1, eps=0.0001, dt=0.01,
                                     plotcircles=2, betagiv=beta))


    ### Below can be used to create the plots showing affect of beta on eigenvalues
    # figP = plt.figure()
    # plt.plot(betaVec, mineigsP, '^--', c='coral')
    # plt.plot(betaVec, maxeigsP, '+--', c='lime')
    # plt.plot(betaVec, np.abs(mineigsP - maxeigsP), 'o--', c='aqua')
    # plt.legend(["Minimum Eigenvalue", "Maximum Eigenvalue", "abs(Max - Min) Eigenvalue"])
    # # plt.title("Eigenvalue change for gamma = 1,2,3 with stab-method = Tang(2016)")
    # plt.title("Eigenvalue change for beta = 0.1 - 1.0 with stab-method = Velo")
    # plt.xlabel("beta")
    # # plt.ylabel("")
    # plt.show()
    # figP.savefig('compare/' + 'Beta_1_thru_0_velo' + '.png')

    # level:
    #   1:  coarse grid h = 0.05
    #   2:  medium grid h = 0.025

    # bfs:
    #   1:  (u.n)(u.v)
    #   2:  |u.n|_(u.v)  inertial
    #   3:  tangential
    #   4:  2016 tangential

