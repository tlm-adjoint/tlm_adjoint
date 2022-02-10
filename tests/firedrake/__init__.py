import pytest
pytest.importorskip("firedrake", reason="Firedrake not available")
del pytest
