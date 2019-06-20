from dolfin import *
import numpy as np


def abs_n(x):
    return 0.5*(x - abs(x))


def nse(Re=1000, temam=False, bfs=False, level=1,velocity_degree=2, eps=0.0002, dt = 0.001):
    
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
    k = Constant(1/dt)
    mu = Constant(0.035)
    rho = Constant(1.2)
    eps = Constant(eps)

    U = 3/2*0.5*Re*float(mu)/float(rho)

    print('\n Re = {}, U = {}\n'.format(Re, U))

    u_ = theta*u + (1 - theta)*u0
    theta_p = theta
    p_ = theta_p*p + (1 - theta_p)*p0

    F = (
        k*rho*dot(u - u0, v)*dx
        + mu*inner(grad(u_), grad(v))*dx
        - p_*div(v)*dx + q*div(u)*dx
        + rho*dot(grad(u_)*u0, v)*dx
    )

    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    
    # eps = Constant(dt*float(mu)/(0.1**2*float(rho)))

    if temam:
        F += 0.5*rho*div(u0)*dot(u_, v)*dx

    if bfs == 1:
        F -= 0.5*rho*dot(u0, n)*dot(u_, v)*ds(2)
    elif bfs == 2:
        F -= 0.5*rho*abs_n(dot(u0, n))*dot(u_, v)*ds(2)
    elif bfs == 3:
        Ctgt = h**2
        F -= Ctgt*0.5*rho*abs_n(dot(u0, n))*(
            Dx(u[0], 1)*Dx(v[0], 1) + Dx(u[1], 1)*Dx(v[1], 1))*ds(2)


    elif velocity_degree == 1 and float(eps):
        F += eps/mu*h**2*inner(grad(p_), grad(q))*dx

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
    
    xdmf_u = XDMFFile('results/u_' + suf + '.xdmf')
    xdmf_p = XDMFFile('results/p_' + suf + '.xdmf')
    xdmf_tau = XDMFFile('tau_sd_' + suf + '.xdmf')
    # xdmf_u.parameters['rewrite_function_mesh'] = False
    # xdmf_p.parameters['rewrite_function_mesh'] = False
    # xdmf_tau.parameters['rewrite_function_mesh'] = False

    u0.rename('u', 'u')
    p0.rename('p', 'p')

    T = 0.4
    
    for t in np.arange(dt, T+dt, dt):
        print('t = {}'.format(t))
        inflow.t = t
        assemble(a, tensor=A)
        b = assemble(L)
        [bc.apply(A, b) for bc in bcs]
        solve(A, w.vector(), b)
        #print('|u|:', norm(u0))
        #print('|p|:', norm(p0))
        #print('div(u):', assemble(div(u0)*dx))
        xdmf_u.write(u0, t)
        xdmf_p.write(p0, t)

    del xdmf_u, xdmf_p


if __name__ == '__main__':
    Re = [2000]
    
    for Re_ in Re:
        nse(Re_, level=1, temam=True, bfs=1, velocity_degree=1, eps=0.0001, dt=0.01)

    # level:
    #   1:  coarse grid h = 0.05
    #   2:  medium grid h = 0.025


    # bfs:
    #   1:  (u.n)(u.v)
    #   2:  |u.n|_(u.v)  inertial
    #   3:  tangential
