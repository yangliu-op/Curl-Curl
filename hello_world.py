from firedrake import *
from firedrake.petsc import PETSc
print = PETSc.Sys.Print

bad = 3
nx = 10
mesh = UnitSquareMesh(nx, nx)

degree = 1
V = FunctionSpace(mesh, "N1curl", degree)

uh = Function(V, name="solution")
v = TestFunction(V)
u = TrialFunction(V)

x, y = SpatialCoordinate(mesh)
f_expr = as_vector([sin(x)*sin(y), cos(x)*cos(y)])
f = Function(V, name="rhs").interpolate(f_expr)


F = (inner(curl(v), curl(uh))
     + inner(v, f)
    )*dx
    
Jp = (inner(curl(v), curl(u))
     + inner(v, u)
    )*dx

bcs = DirichletBC(V, 0, "on_boundary")

sp = {
    "snes_type": "ksponly",
    "ksp_type": "minres",
    "ksp_monitor": None,
    #"pc_type": "none",
    "ksp_norm_type": "preconditioned",
}


print("Dimension", V.dim())
problem = NonlinearVariationalProblem(F, uh, bcs=bcs, Jp=Jp)
#problem = NonlinearVariationalProblem(F, uh, Jp=Jp)
solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
solver.solve()

File("output/curl.pvd").write(uh, f)
