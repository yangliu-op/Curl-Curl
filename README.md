# Curl Curl u = f

This is a FireDrake + PETSc implementation for a Curl Curl PDE solving.

The Curl Curl operator may contain infinitely many solutions for a general right-hand side. We would like to obtain the pseudo-inverse solution among them. 
We use the lifting formula (described in the paper [Obtaining Pseudo-inverse Solutions With MINRES](https://arxiv.org/abs/2309.17096)) to recover the pseudo-inverse solution for the symmetric least-squares with MINRES solver.

One can drop the Boundry condition or use preconditioners inside the code manunally. 

A corresponding recorded video (no precondioners, with boundry conditions) can be found at [https://www.youtube.com/watch?v=ivRa-O9DCMI](https://www.youtube.com/watch?v=ivRa-O9DCMI). 
This video shows the lifted MINRES iterates converge to the pseudo-inverse solution iteratively.
