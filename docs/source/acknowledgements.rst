References and acknowledgements
===============================

Citing tlm_adjoint
------------------

tlm_adjoint is described in

- James R. Maddison, Daniel N. Goldberg, and Benjamin D. Goddard, 'Automated
  calculation of higher order partial differential equation constrained
  derivative information', SIAM Journal on Scientific Computing, 41(5), pp.
  C417--C445, 2019, doi: 10.1137/18M1209465

The automated assembly and linear solver caching applied by tlm_adjoint is
based on the approach described in

- J. R. Maddison and P. E. Farrell, 'Rapid development and adjoining of
  transient finite element models', Computer Methods in Applied Mechanics and
  Engineering, 276, 95--121, 2014, doi: 10.1016/j.cma.2014.03.010

Checkpointing with tlm_adjoint is described in

- James R. Maddison, 'Step-based checkpointing with high-level algorithmic
  differentiation', Journal of Computational Science 82, 102405, 2024,
  doi: 10.1016/j.jocs.2024.102405

References
----------

dolfin-adjoint
``````````````

tlm_adjoint implements high-level algorithmic differentiation using an
approach based on that used by dolfin-adjoint, described in

- P. E. Farrell, D. A. Ham, S. W. Funke, and M. E. Rognes, 'Automated
  derivation of the adjoint of high-level transient finite element programs',
  SIAM Journal on Scientific Computing 35(4), pp. C369--C393, 2013,
  doi: 10.1137/120873558

tlm_adjoint was developed from a custom extension to dolfin-adjoint.

Taylor remainder convergence testing
````````````````````````````````````

The functions in `tlm_adjoint/verification.py
<autoapi/tlm_adjoint/verification/index.html>`_ implement Taylor remainder
convergence testing using the approach described in

- P. E. Farrell, D. A. Ham, S. W. Funke, and M. E. Rognes, 'Automated
  derivation of the adjoint of high-level transient finite element programs',
  SIAM Journal on Scientific Computing 35(4), pp. C369--C393, 2013,
  doi: 10.1137/120873558

Differentiating fixed-point problems
````````````````````````````````````

The `FixedPointSolver` class in `tlm_adjoint/fixed_point.py
<autoapi/tlm_adjoint/fixed_point/index.html>`_ derives tangent-linear and
adjoint information using the approach described in

- Jean Charles Gilbert, 'Automatic differentiation and iterative processes',
  Optimization Methods and Software, 1(1), pp. 13--21, 1992,
  doi: 10.1080/10556789208805503
- Bruce Christianson, 'Reverse accumulation and attractive fixed points',
  Optimization Methods and Software, 3(4), pp. 311--326, 1994,
  doi: 10.1080/10556789408805572

Binomial checkpointing
``````````````````````

The `MultistageCheckpointSchedule` class in
`tlm_adjoint/checkpoint_schedules/binomial.py
<autoapi/tlm_adjoint/checkpoint_schedules/binomial/index.html>`_ implements the
binomial checkpointing strategy described in

- Andreas Griewank and Andrea Walther, 'Algorithm 799: revolve: an
  implementation of checkpointing for the reverse or adjoint mode of
  computational differentiation', ACM Transactions on Mathematical Software,
  26(1), pp. 19--45, 2000, doi: 10.1145/347837.347846

The `MultistageCheckpointSchedule` class determines a memory/disk storage
distribution using an initial run of the checkpoint schedule, leading to a
distribution equivalent to that in

- Philipp Stumm and Andrea Walther, 'MultiStage approaches for optimal offline
  checkpointing', SIAM Journal on Scientific Computing, 31(3), pp. 1946--1967,
  2009, doi: 10.1137/080718036

The `TwoLevelCheckpointSchedule` class in
`tlm_adjoint/checkpoint_schedules/binomial.py
<autoapi/tlm_adjoint/checkpoint_schedules/binomial/index.html>`_ implements the
two-level mixed periodic/binomial checkpointing approach described in

- Gavin J. Pringle, Daniel C. Jones, Sudipta Goswami, Sri Hari Krishna
  Narayanan, and Daniel Goldberg, 'Providing the ARCHER community with adjoint
  modelling tools for high-performance oceanographic and cryospheric
  computation', version 1.1, EPCC, 2016

and in the supporting information for

- D. N. Goldberg, T. A. Smith, S. H. K. Narayanan, P. Heimbach, and M.
  Morlighem,, 'Bathymetric influences on Antarctic ice-shelf melt rates',
  Journal of Geophysical Research: Oceans, 125(11), e2020JC016370, 2020,
  doi: 10.1029/2020JC016370

PyTorch
```````

The PyTorch interface in `tlm_adjoint/torch.py
<autoapi/tlm_adjoint/torch/index.html>`_ follows the same principles as
described in

    - Nacime Bouziani and David A. Ham, 'Physics-driven machine learning models
      coupling PyTorch and Firedrake', 2023, arXiv:2303.06871v3
     
Funding
-------

Early development work leading to tlm_adjoint was conducted as part of a U.K.
Natural Environment Research Council funded project (NE/L005166/1). Further
development has been conducted as part of a U.K. Engineering and Physical
Sciences Research Council funded project (EP/R021600/1) and a Natural
Environment Research Council funded project (NE/T001607/1).
