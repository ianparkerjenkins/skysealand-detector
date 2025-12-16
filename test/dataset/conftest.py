# tests/conftest.py
import pytest

from skysealand import logging_setup


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    logging_setup.setup_logging()
