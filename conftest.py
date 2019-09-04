#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def pytest_configure(config):
    config.addinivalue_line("markers", "examples: example scripts")
    config.addinivalue_line("markers", "fenics: FEniCS tests")
    config.addinivalue_line("markers", "firedrake: Firedrake tests")
    config.addinivalue_line("markers", "numpy: NumPy tests")
