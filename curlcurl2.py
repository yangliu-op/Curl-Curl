from firedrake import *
from firedrake.petsc import PETSc
print = PETSc.Sys.Print
niter = 500

nx = 100
mesh = UnitSquareMesh(nx, nx)
x, y = SpatialCoordinate(mesh)

data = 2
if data == 1:
    psi = cos(4*pi*x) * cos(4*pi*y)
    bc_subdomain = ["on_boundary"]
elif data == 2:
    psi = sin(4*pi*x) * sin(4*pi*y)
    bc_subdomain = []


degree = 1
V = FunctionSpace(mesh, "N1curl", degree)

u_expr = perp(grad(psi))
u_exact = Function(V, name="Exact_Sol")
u_exact.interpolate(u_expr)


bcs = [DirichletBC(V, 0, sub) for sub in bc_subdomain]

uh = Function(V, name="solution")
v = TestFunction(V)
u = TrialFunction(V)





Q = FunctionSpace(mesh, "Lagrange", degree)
# PCG64 random number generator
pcg = PCG64(seed=5)
rg = RandomGenerator(pcg)
scale = 5E1
# beta distribution
# phi_noise = rg.beta(Q, 1.0, 3.0)
# # uniform distribution
phi_noise = rg.uniform(Q, -scale, scale)
phi_bc = [DirichletBC(Q, 0, sub) for sub in bc_subdomain]
for bc in phi_bc:
    bc.zero(phi_noise)
# f_noise = assemble(inner(grad(v), phi_noise)*dx, bcs=bcs)
f_noise = assemble(inner(v, grad(phi_noise))*dx, bcs=bcs)




a = inner(curl(u), curl(v)) * dx
L = (a(v, u_exact)
     + action(f_noise, v)
    )
Jp = (inner(curl(u), curl(v))
     + inner(u, v)
    )*dx


sp = {
    "snes_type": "ksponly",
    # "ksp_type": "lsqr",
    "ksp_type": "minres",
    "ksp_max_it": niter,
    "ksp_convergence_test": "skip",
    "ksp_monitor": None,
    "pc_type": "none",
    "ksp_norm_type": "preconditioned",
}


print("Dimension", V.dim())
problem = LinearVariationalProblem(a, L, uh, bcs=bcs, aP=Jp)
solver = LinearVariationalSolver(problem, solver_parameters=sp, options_prefix="")
solver.solve()


# Ax = curl(curl(uh))
# r = f - Ax
r = assemble(problem.F, bcs=bcs)
# L2 Lebesgue inner product (Hilbert space)
# l2 Euclidean inner product (R^n)
rstar = r.riesz_representation(riesz_map="l2", bcs=bcs)
rstar.rename("RHS")

# lft = uh - inner(r, uh)/inner(r, rstar) * rstar
c = assemble(action(r, uh)) / assemble(action(r, rstar))
ulft = Function(V, name="Lifted_MINRES")
ulft.assign(uh - c * rstar)


sol = Function(V, name="MINRES")
sol.assign(uh)

f_star = f_noise.riesz_representation("l2")
f_star.rename("noise")
File("output/curl.pvd").write(f_star, rstar, sol, ulft, u_exact)


udiff = Function(V, name="difference")
udiff.assign(rstar)
with udiff.dat.vec_ro as uv:
	print("RHS norm", uv.norm())
# 
    
udiff = Function(V, name="difference")
udiff.assign(u_exact)
with udiff.dat.vec_ro as uv:
	print("Exact norm", uv.norm())
# 
    
udiff = Function(V, name="difference")
udiff.assign(ulft)
with udiff.dat.vec_ro as uv:
	print("ulft norm", uv.norm())

udiff = Function(V, name="difference")
udiff.assign(uh)
with udiff.dat.vec_ro as uv:
	print("u_minres norm", uv.norm())

udiff = Function(V, name="difference")
udiff.assign(uh - u_exact)
with udiff.dat.vec_ro as uv:
	print("u_exact - u_minres norm", uv.norm())
	

udiff.assign(ulft - u_exact)
with udiff.dat.vec_ro as uv:
	print("u_exact - u_lifted norm", uv.norm())


uh.assign(ulft)
rlft = assemble(problem.F, bcs=bcs)
with r.dat.vec_ro as rv, rlft.dat.vec_ro as rlftv:
	#rv.view()
	print("MINRES residual norm", rv.norm())
	print("Lifted residual norm", rlftv.norm())
	
# Bcs
# RHS norm 6809.19331120761
# Exact norm 8.907952592193048
# ulft norm 9.556779874640705
# u_minres norm 2150.5537316259115
# u_exact - u_minres norm 2150.5354143619934
# u_exact - u_lifted norm 3.5418693318190058
# MINRES residual norm 6809.19331120761
# Lifted residual norm 6813.504900263961
    
# NO bcs
# RHS norm 6854.490638257576
# Exact norm 8.907952592193048
# ulft norm 7.942369959386413
# u_minres norm 1111.84435340314
# u_exact - u_minres norm 1111.8341333175497
# u_exact - u_lifted norm 4.076202449621155
# MINRES residual norm 6854.490638257576
# Lifted residual norm 6854.491594938664


