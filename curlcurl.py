from firedrake import *
from firedrake.petsc import PETSc
print = PETSc.Sys.Print

nx = 100
mesh = UnitSquareMesh(nx, nx)
x, y = SpatialCoordinate(mesh)

data = 2
if data == 1:
    psi = cos(2*pi*x) * cos(2*pi*y)
    bc_subdomain = ["on_boundary"]
elif data == 2:
    psi = sin(2*pi*x) * sin(2*pi*y)
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
scale = 1
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
Jp = a + inner(u, v) * dx
# Jp = inner(curl(u), curl(v)) * dx + inner(u, v) * dx


preconditioner = True
# preconditioner = False
if preconditioner:
    pc_type = "cholesky"
else:
    pc_type = "none"


sp = {
    "snes_type": "ksponly",
    # "ksp_type": "lsqr",
    "ksp_type": "minres",
    "ksp_max_it": 1000,
    "ksp_convergence_test": "skip",
    "ksp_monitor": None,
    "pc_type": pc_type,
    "ksp_norm_type": "preconditioned",
    "ksp_minres_nutol": 1E-1,
    # "ksp_minres_lifting": True, # we are going to add this in the future
}


print("Dimension", V.dim())
problem = LinearVariationalProblem(a, L, uh, bcs=bcs, aP=Jp)
solver = LinearVariationalSolver(problem, solver_parameters=sp, options_prefix="")
solver.solve()

def riesz_map(functional):
    function = Function(functional.function_space().dual())
    with functional.dat.vec as x, function.dat.vec as y:
        solver.snes.ksp.pc.apply(x, y)
    return function

if not preconditioner:
    # L2 Lebesgue inner product (Hilbert space)
    # l2 Euclidean inner product (R^n)
    riesz_map = "l2"

# Ax = curl(curl(uh))
# r = f - Ax
r = assemble(problem.F, bcs=bcs)
rstar = r.riesz_representation(riesz_map=riesz_map, bcs=bcs)
rstar.rename("RHS")

# lft = uh - inner(r, uh)/inner(r, rstar) * rstar
c = assemble(action(r, uh)) / assemble(action(r, rstar))
ulft = Function(V, name="Lifted_MINRES")
ulft.assign(uh - c * rstar)


sol = Function(V, name="MINRES")
sol.assign(uh)

f_star = f_noise.riesz_representation(riesz_map=riesz_map)
f_star.rename("noise")
File("output/curl.pvd").write(f_star, rstar, sol, ulft, u_exact)


udiff = Function(V, name="difference")
udiff.assign(rstar)
with udiff.dat.vec_ro as uv:
	print("RHS l2-norm", uv.norm())
#

udiff = Function(V, name="difference")
udiff.assign(u_exact)
with udiff.dat.vec_ro as uv:
	print("Exact l2-norm", uv.norm())
#

udiff = Function(V, name="difference")
udiff.assign(ulft)
with udiff.dat.vec_ro as uv:
	print("ulft l2-norm", uv.norm())

udiff = Function(V, name="difference")
udiff.assign(uh)
with udiff.dat.vec_ro as uv:
	print("u_minres norm", uv.norm())

udiff = Function(V, name="difference")
udiff.assign(uh - u_exact)
print("u_exact - u_minres M-norm", sqrt(assemble(Jp(udiff, udiff))))
with udiff.dat.vec_ro as uv:
	print("u_exact - u_minres l2-norm", uv.norm())


udiff.assign(ulft - u_exact)
print("u_exact - u_lifted M-norm", sqrt(assemble(Jp(udiff, udiff))))
with udiff.dat.vec_ro as uv:
	print("u_exact - u_lifted l2-norm", uv.norm())


uh.assign(ulft)
rlft = assemble(problem.F, bcs=bcs)
with r.dat.vec_ro as rv, rlft.dat.vec_ro as rlftv:
	#rv.view()
	print("MINRES residual l2-norm", rv.norm())
	print("Lifted residual l2-norm", rlftv.norm())
