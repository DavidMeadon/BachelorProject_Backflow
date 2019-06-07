from dolfin import *
import numpy as np
import matplotlib.pyplot as plt


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


class MyParameter:
    param = 1

    def __init__(self, value):
        self.param = value

    def update(self, value):
        self.param = value

    def value(self):
        return self.param


def nse(Re=1000, temam=False, bfs=False, level=1, velocity_degree=2, eps=0.0002, dt=0.001):
    mesh_root = 'stenosis_f0.6'
    if level == 2:
        mesh_root += '_fine'

    mesh = Mesh(mesh_root + '.xml')
    boundaries = MeshFunction('size_t', mesh, mesh_root + '_facet_region.xml')
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

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

    F = (
            k * rho * dot(u - u0, v) * dx
            + mu * inner(grad(u_), grad(v)) * dx
            - p_ * div(v) * dx + q * div(u) * dx
            + rho * dot(grad(u_) * u0, v) * dx
    )

    n = FacetNormal(mesh)
    h = CellDiameter(mesh)

    # eps = Constant(dt * float(mu) / (0.1 ** 2 * float(rho)))

    if temam:
        F += 0.5 * rho * div(u0) * dot(u_, v) * dx

    beta = Constant(1)  # TODO add an initial beta value, probably 1

    ### Added here beta parameter to the stabilization terms which we can control
    if bfs == 1:
        F -= 0.5 * rho * beta * dot(u0, n) * dot(u_, v) * ds(2)
    elif bfs == 2:
        F -= 0.5 * rho * beta * abs_n(dot(u0, n)) * dot(u_, v) * ds(2)
    elif bfs == 3:
        Ctgt = h ** 2
        F -= Ctgt * 0.5 * rho * abs_n(dot(u0, n)) * (
                Dx(u[0], 1) * Dx(v[0], 1) + Dx(u[1], 1) * Dx(v[1], 1)) * ds(2)


    elif velocity_degree == 1 and float(eps):
        F += eps / mu * h ** 2 * inner(grad(p_), grad(q)) * dx

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
    #     print(type(F))
    # MAIN SOLVING LOOP
    for t in np.arange(dt, T + dt, dt):
        print('t = {}'.format(t))

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
    plt.figure()
    # plt.plot(viscEnergyVec, 'b', label="Viscous")
    # plt.plot(ToteviscEnergyVec, 'lightseagreen', label="tot Viscous")
    plt.plot(incEnergyVec, 'coral', label="Incoming")
    #     plt.plot(numEnergyVec, 'y', label="Numerical")
    #     plt.plot(stabEnergyVec, 'g', label='Stabilization')
    plt.plot(ToteviscEnergyVec + stabEnergyVec + numEnergyVec, 'deepskyblue', label='Total corrective energy')
    plt.legend(loc='upper left')
    plt.show()

    # del xdmf_u, xdmf_p


if __name__ == '__main__':
    Re = [2000]

    for Re_ in Re:
        nse(Re_, level=1, temam=True, bfs=2, velocity_degree=1, eps=0.0001, dt=0.01)

        ## Weird results for the stabilization if bfs = 2, and doenst work for bfs = 3

    # level:
    #   1:  coarse grid h = 0.05
    #   2:  medium grid h = 0.025

    # bfs:
    #   1:  (u.n)(u.v)
    #   2:  |u.n|_(u.v)  inertial
    #   3:  tangential

