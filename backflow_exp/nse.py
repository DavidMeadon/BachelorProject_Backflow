from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.sparse.linalg as ssl
from numpy import linalg as LA
from math import fabs
import progress as prog

# plt.rcParams['animation.ffmpeg_path'] = '/snap/bin/ffmpeg'




def abs_n(x):
    return -0.5 * (x - abs(x))


# TODO rework the backflowarea function since it doesn't do as it should.
def backflowarea(a, b, c):
    if assemble(abs_n(dot(a, b)) * c) > 0:
        return 1
    else:
        return 0


def test(a, b):
    print(dot(a, b))
    return 1


def betaupdate(current_sol, prev_sol, facet_norm, dx, beta):
    if assemble(abs_n(dot(current_sol, facet_norm)) * div(current_sol) * dx) > assemble(abs_n(dot(prev_sol, facet_norm)) * div(prev_sol) * dx):
        return min(max(assemble(beta * dx) + 0.1, 0.2), 1)
    else:
        return min(max(assemble(beta * dx) - 0.1, 0.2), 1)


def radfinder(M, k, n):
    tote = 0
    for p in range(n):
        if p != k:
            tote += np.abs(M[p])
    return tote


def gershgorinplotter(mat, t, bfsm):
    n = mat.shape[1]
    # radii = np.zeros((n, 1))
    # centers = np.zeros((n,1))
    fig = plt.figure()
    axes = plt.gca()
    axes.set_xlim([-10, 10])
    axes.set_ylim([-10, 10])
    if n > 0:
        for k in range(n):
            # centers[k] = mat[k, k]
            center = mat[k, k]
            rad = radfinder(mat[k, :], k, n)
            # radii[k] = radfinder(mat[k, :], k, n)
            circle1 = plt.Circle((center, 0), rad)
            plt.gcf().gca().add_artist(circle1)
    plt.title('Method: ' + bfsm + ', Time: ' + str(t))
    return fig

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

def plotCircles(circles, t, bfsm, beta):
    fig, ax = plt.subplots()
    plt.title('Method: ' + bfsm + ', Time: ' + str(t))
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    if circles == []:
        return fig
    index, radi = zip(*circles)
    Xupper = 0.3#max(index) + np.std(index)
    Xlower = -0.1#min(index) - np.std(index)
    Ylimit = max(radi) + np.std(index)
    ax = plt.gca()
    ax.cla()
    ax.set_xlim((Xlower,Xupper))
    ax.set_ylim((-Ylimit,Ylimit))
    plt.title('Method: ' + bfsm + ', Time: ' + str(t) + ', Beta: ' + str(beta))
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    for x in range(0,len(circles)):
        circ = plt.Circle((index[x],0), radius = radi[x])
        ax.add_artist(circ)
    ax.plot([Xlower,Xupper],[0,0],'k--')
    ax.plot([0,0],[-Ylimit,Ylimit],'k--')
    return fig

# plt.show()

class MyParameter:
    param = 1

    def __init__(self, value):
        self.param = value

    def update(self, value):
        self.param = value

    def value(self):
        return self.param


def nse(Re=1000, temam=False, bfs=False, level=1, velocity_degree=2, eps=0.0002, dt=0.001):
    # pbar = ProgressBar()
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

    theta = Constant(1)
    k = Constant(1 / dt)
    mu = Constant(0.035)
    rho = Constant(1.2)
    eps = Constant(eps)

    U = 3 / 2 * 0.5 * Re * float(mu) / float(rho)

    print('\n Re = {}, U = {}\n'.format(Re, U))

    u_ = theta * u + (1 - theta) * u0
    theta_p = theta
    p_ = theta_p * p + (1 - theta_p) * p0

    laplace = mu * inner(grad(u_), grad(v)) * dx

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

    beta = Constant(1) #TODO add an initial beta value, probably 1

    backflow_func = 0.5 * rho * abs_n(dot(u0, n)) * dot(u_, v) * ds(2) #0.5 * rho * abs_n(div(u0)) * dot(u_, v) * dx



    G = 0
    ### Added here beta parameter to the stabilization terms which we can control
    if bfs == 1:
        stabmethod = 'velocity-penalization'
        G = 0.5 * rho * beta * dot(u0, n) * dot(u_, v) * ds(2)
        F -= G
    elif bfs == 2:
        stabmethod = 'velocity-penalization negative part'
        G = 0.5 * rho * beta * abs_n(dot(u0, n)) * dot(u_, v) * ds(2)
        F -= G
    elif bfs == 3:
        stabmethod = 'velocity gradient penalization'
        Ctgt = h ** 2
        G = Ctgt * 0.5 * rho * abs_n(dot(u0, n)) * (
                Dx(u[0], 1) * Dx(v[0], 1) + Dx(u[1], 1) * Dx(v[1], 1)) * ds(2)
        F -= G
    elif velocity_degree == 1 and float(eps):
        F += eps / mu * h ** 2 * inner(grad(p_), grad(q)) * dx

    testfunc = laplace - backflow_func + G

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

    suf = 'bfs{}_tem{}_Re{}'.format(int(bfs), int(temam), Re)
    if velocity_degree == 1:
        suf = 'p1_' + suf
    suf = 'l{}_'.format(level) + suf

    # xdmf_u = XDMFFile('results/u_' + suf + '.xdmf')
    # xdmf_p = XDMFFile('results/p_' + suf + '.xdmf')
    # xdmf_tau = XDMFFile('tau_sd_' + suf + '.xdmf')
    # xdmf_u.parameters['rewrite_function_mesh'] = False
    # xdmf_p.parameters['rewrite_function_mesh'] = False
    # xdmf_tau.parameters['rewrite_function_mesh'] = False
    #
    # u0.rename('u', 'u')
    # p0.rename('p', 'p')

    w0 = Function(W)
    r0, s0 = w0.split()

    w1 = Function(W)
    r1, s1 = w1.split()

    # FINAL TIME
    T = 0.4
    ite = 0
    # PLOTTING VECTORS
    viscEnergyVec = np.zeros(((int)(T / dt), 1))
    ToteviscEnergyVec = np.zeros(((int)(T / dt), 1))
    incEnergyVec = np.zeros(((int)(T / dt), 1))
    incEnergyVec2 = np.zeros(((int)(T / dt), 1))
    numEnergyVec = np.zeros(((int)(T / dt), 1))
    stabEnergyVec = np.zeros(((int)(T / dt), 1))
    # avgeig = np.zeros(((int)(T / dt), 1))
    #     print(type(F))

    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    # anitemp = []
    # print(testfunc)
    pt = prog.progress_timer(description='Time Iterations', n_iter=40)
    # MAIN SOLVING LOOP
    for t in np.arange(dt, T + dt, dt):
        # print('t = {}'.format(round(t, 2)))

        #### May not be necessary ####
        w0.assign(w)
        r0, s0 = w0.split()
        ####

        inflow.t = t
        assemble(a, tensor=A)
        b = assemble(L)
        [bc.apply(A, b) for bc in bcs]
        solve(A, w.vector(), b)

        #### May not be necessary ####

        ###


        w1.assign(w)
        r1, s1 = w1.split()

        r1.vector().set_local(u0.vector().get_local() * (u0.vector().get_local() < 0))

        # ite +=1
        # if ite==5:
        #    plt.plot(r1.vector().get_local())
        #    plt.plot(u0.vector().get_local())

        # print('|u|:', norm(u0))
        # print('|p|:', norm(p0))
        # print('div(u):', assemble(div(u0)*dx))

        ### This was trying to view U0 as a numpy array but didnt show anything meaningful ####
        # for i in u0.vector().get_local():
        #     print(i)
        # print(u0.vector().get_local())

        # BACKFLOW KINETIC ENERGY CHANGE
        BKE = assemble((rho / 2) * abs_n(dot(r0, n)) * dot(u0, u0) * ds(2))

        # BACKFLOW VISCOUS ENERGY CHANGE
        # BVE = assemble(mu * inner(grad(r1),grad(r1)) * ds(2))
        # TVE = assemble(mu * inner(grad(u0),grad(u0))  * ds(2))
        TVE = assemble(mu * inner(grad(u0), grad(u0)) * dx)

        #         print( (abs_n(u0) != 0) )
        # BVEs = assemble(mu * np.abs(div(u0)) * div(r1) * ds(2))

        # TODO rework the backflowarea function so that it is one when there is backflow and 0 otherwise

        # ADDING TO VECTORS
        # viscEnergyVec[(int)(t / dt) - 1] = BVE
        ToteviscEnergyVec[(int)(t / dt) - 1] = TVE

        incEnergyVec[(int)(t / dt) - 1] = BKE

        numEnergyVec[(int)(t / dt) - 1] = assemble((dot(u0 - r0, u0 - r0)) * ds(2))

        ### Here want to calculate how much energy chnages due to the stabilization
        if bfs == 1:
            stabEnergyVec[(int)(t / dt) - 1] = assemble(0.5 * beta * rho * abs_n(dot(u0, n)) * dot(u0, u0) * ds(2))
        elif bfs == 2:
            stabEnergyVec[(int)(t / dt) - 1] = assemble(0.5 * beta * rho * abs_n(dot(u0, n)) * dot(u0, u0) * ds(2))
        elif bfs == 3:
            Ctgt = h ** 2
            stabEnergyVec[(int)(t / dt) - 1] = assemble(Ctgt * 0.5 * rho * abs_n(dot(u0, n)) * (
                    Dx(u0[0], 1) * Dx(u0[0], 1) + Dx(u0[1], 1) * Dx(u0[1], 1)) * ds(2))

        ## Trying eigenvalue stuff - Quite slow and expensive
        # MAT = assemble(lhs(G))
        # VEC = np.array(MAT.array())
        #
        #
        # eigvals = eigs(VEC, k=100, M=None, sigma=None, which='LM', v0=None, ncv=None, maxiter=None, tol=0,
        #                return_eigenvectors=False, Minv=None, OPinv=None, OPpart=None)
        #
        # avgeig[(int)(t / dt) - 1] = np.mean(eigvals)

        ## Finding backflow region of Matrix
        lap = assemble(lhs(laplace))#testfunc))
        lap2 = lap
        for bc in bcs: bc.apply(lap)
        print(LA.norm((np.array(lap.array()) - np.array(lap2.array()))))
        lapmat = np.array(lap.array())
        lapmat2 = sp.sparse.bsr_matrix(lap.array())
        # print("Finding Backflow region")
        backflow_mat = assemble(lhs(backflow_func))
        backflow_vec = np.array(backflow_mat.array())
        reduced_laplace = lapmat*(backflow_vec != 0)
        laplace_backflow = reduced_laplace[~(reduced_laplace == 0).all(1)]
        laplace_backflow_final = np.transpose(laplace_backflow.transpose()[~(laplace_backflow.transpose() == 0).all(1)])

        # anitemp.append(laplace_backflow_final)
        # fig = gershgorinplotter(laplace_backflow_final, round(t, 2), stabmethod)
        # print("Making Circles: " + str(len(laplace_backflow_final)))
        circles = GregsCircles(laplace_backflow_final)
        fig = plotCircles(circles, round(t, 2), stabmethod, assemble(beta*ds(2)))
        # print("Finding Eigenvalues")
        smalleig = ssl.eigs(lapmat2, 5, sigma=1e-6, which='LM', return_eigenvectors=False)
        print(smalleig)
        eigenvals = LA.eigvals(laplace_backflow_final)
        for eigval in eigenvals:
            plt.plot(eigval.real, eigval.imag, 'r+')
        fig.savefig('circles/' + str(round(t*100)) + 'gersh.png')
        plt.close(fig)
        print(eigenvals)
        del lap, lapmat, backflow_mat, backflow_vec, reduced_laplace, laplace_backflow, laplace_backflow_final, eigenvals
        # eigvals, eigvec = LA.eig(laplace_backflow_final)

        ##Gershgorin stuff:

        # fig = plt.figure()
        #
        # fig = gershgorinplotter(lapmat[-500:][:, -500:])
        # anitemp.append(fig)



        ### Print out the values for the energy changes at the current time step
        # print('Viscous energy change:', viscEnergyVec[(int)(t/dt) - 1])
        # print('Incoming energy change:', incEnergyVec[(int)(t/dt) - 1])
        # print('Numerical energy:', numEnergyVec[(int)(t/dt) - 1])

        # xdmf_u.write(u0, t)
        # xdmf_p.write(p0, t)
        # xdmf_p.write(p0, t)

        # print(F)
        #         if bfs == 1:
        #           F += 0.5 * rho * beta * dot(u0, n) * dot(u_, v) * ds(2)
        #         elif bfs == 2:
        #           F += 0.5 * rho * beta * abs_n(dot(u0, n)) * dot(u_, v) * ds(2)
        #         elif bfs == 3:
        #             Ctgt = h ** 2
        #             F += Ctgt * 0.5 * rho * abs_n(dot(u0, n)) * (
        #                 Dx(u[0], 1) * Dx(v[0], 1) + Dx(u[1], 1) * Dx(v[1], 1)) * ds(2)
        # beta.assign(betaupdate(u0, r1, n, ds(2), beta))
        # print(assemble(beta*ds(2)))
        #         beta = betaupdate(u0, r1, n, ds(2), beta) #This isn't updating the one in the function
        # #         print(assemble(beta*ds(2)))

        #         if bfs == 1:
        #           F -= 0.5 * rho * beta * dot(u0, n) * dot(u_, v) * ds(2)
        #         elif bfs == 2:
        #           F -= 0.5 * rho * beta * abs_n(dot(u0, n)) * dot(u_, v) * ds(2)
        #         elif bfs == 3:
        #             Ctgt = h ** 2
        #             F -= Ctgt * 0.5 * rho * abs_n(dot(u0, n)) * (
        #                 Dx(u[0], 1) * Dx(v[0], 1) + Dx(u[1], 1) * Dx(v[1], 1)) * ds(2)

        #         a = lhs(F)
        #         L = rhs(F)

        #         A = assemble(a)

        # print(F)
        # beta = beta/2;    # TODO change the beta parameter to update every iteration of the loop
        # Is it possible to solve along a specific boundary?
        # Can we get the matrices use behind the scenes in fenics?
        # print(F)
        pt.update()
    pt.finish()


    ## Animating the Gershgorin circles
    # fig, ax = plt.subplots()
    #
    # def animate(i):
    #     animat = anitemp[i]
    #     centers, radii = gershgorinplotter(animat)
    #     for idx in range(centers.size):
    #         circle1 = plt.Circle(centers[idx], radii[idx])
    #         fig.gca().add_artist(circle1)
    #
    # #
    # # ani = FuncAnimation(fig, animate, frames=40, repeat=True)\
    # ani = animation.FuncAnimation(fig, animate, np.arange(0, 40, 1))
    # plt.show()
    #
    # ani.save("Gershgorin.mp4")

    # plt.figure()
    # plt.plot(avgeig)
    # plt.title('Average Eigenvalues of ' + stabmethod)
    # plt.show()


    # plt.figure()
    # # plt.plot(viscEnergyVec, 'b', label="Viscous")
    # # plt.plot(ToteviscEnergyVec, 'forestgreen', label="tot Viscous")
    # plt.plot(incEnergyVec, 'red', label="Incoming")
    # # plt.plot(numEnergyVec, 'yellow', label="Numerical")
    # # plt.plot(stabEnergyVec, 'orange', label='Stabilization')
    # plt.plot(ToteviscEnergyVec + numEnergyVec, 'deepskyblue', label='Total corrective energy')
    # plt.legend(loc='upper left')
    # plt.title('Energy changes of ' + stabmethod)
    # plt.show()

    # del xdmf_u, xdmf_p


if __name__ == '__main__':
    Re = [2000]

    for Re_ in Re:
        nse(Re_, level=1, temam=True, bfs=2, velocity_degree=1, eps=0.0001, dt=0.01)

        ## Weird results for the stabilization if bfs = 2,3. Stabilization Energy is too high

    # level:
    #   1:  coarse grid h = 0.05
    #   2:  medium grid h = 0.025

    # bfs:
    #   1:  (u.n)(u.v)
    #   2:  |u.n|_(u.v)  inertial
    #   3:  tangential

