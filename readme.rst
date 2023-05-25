tlm_adjoint
===========

`tlm_adjoint <https://tlm-adjoint.github.io>`_ is a high-level algorithmic
differentiation tool, principally for use with
`FEniCS <https://fenicsproject.org>`_ or `Firedrake
<https://firedrakeproject.org>`_.

The primary aim of tlm_adjoint is to enable higher order adjoint calculations
– and in particular to compute Hessian information – while also using adjoint
checkpointing schedules, and allowing for caching of assembled finite element
data, and caching of linear solver data.

Installation
------------

tlm_adjoint can be installed using pip, e.g.

.. code-block:: sh

    pip install .

run in the tlm_adjoint root directory.

License
-------

tlm_adjoint is licensed under the GNU LGPL version 3.
