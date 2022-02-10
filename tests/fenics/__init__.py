import pytest
pytest.importorskip("fenics", reason="FEniCS not available")
del pytest
