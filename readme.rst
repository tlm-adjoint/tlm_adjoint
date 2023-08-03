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

About
-----        
:Citation:
    .. image:: https://img.shields.io/badge/DOI-10.1137/18M1209465-blue
        :target: https://doi.org/10.1137/18M1209465
        :alt: SISC paper

    .. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7695475.svg
        :target: https://doi.org/10.5281/zenodo.7695475
        :alt: Zenodo

:Tests:
    .. image:: https://github.com/tlm-adjoint/tlm_adjoint/actions/workflows/test-base.yml/badge.svg?branch=main&event=push
      :target: https://github.com/tlm-adjoint/tlm_adjoint/actions/workflows/test-base.yml
      :alt: Base tests status

    .. image:: https://github.com/tlm-adjoint/tlm_adjoint/actions/workflows/test-fenics.yml/badge.svg?branch=main&event=push
      :target: https://github.com/tlm-adjoint/tlm_adjoint/actions/workflows/test-fenics.yml
      :alt: FEniCS tests status
    
    .. image:: https://github.com/tlm-adjoint/tlm_adjoint/actions/workflows/test-firedrake.yml/badge.svg?branch=main&event=push
      :target: https://github.com/tlm-adjoint/tlm_adjoint/actions/workflows/test-firedrake.yml
      :alt: Firedrake tests status

:License:
    .. image:: https://img.shields.io/badge/license-GNU--LGPL--v3-green
        :target: https://github.com/tlm-adjoint/tlm_adjoint/blob/main/LICENSE
        :alt: GNU LGPL version 3
