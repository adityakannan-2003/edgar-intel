"""Test configuration.

Every test in this suite runs without a database, without a network call, and
without downloading a model. That is a deliberate constraint: a test suite that
needs infrastructure is a test suite that stops being run.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("EDGAR_LLM_PROVIDER", "fake")
os.environ.setdefault("EDGAR_EMBED_PROVIDER", "fake")
os.environ.setdefault("EDGAR_USER_AGENT", "edgar-intel-tests tests@example.com")


@pytest.fixture(autouse=True)
def _reset_caches():
    from edgar_intel.config import reset_settings_cache
    from edgar_intel.providers import reset_provider_cache

    reset_settings_cache()
    reset_provider_cache()
    yield
    reset_settings_cache()
    reset_provider_cache()


@pytest.fixture
def sample_filing_text() -> str:
    return (
        "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\n\n"
        "Table of Contents\n"
        "Item 1. Business\n"
        "Item 1A. Risk Factors\n"
        "Item 7. Management's Discussion and Analysis\n\n"
        "Item 1. Business\n"
        "The Company designs and manufactures products sold worldwide. "
        "It operates in a single reportable segment and relies on a global "
        "network of contract manufacturers for assembly operations.\n\n"
        "Item 1A. Risk Factors\n"
        "The Company depends on a limited number of suppliers for critical "
        "components. Disruption at any single-source supplier could materially "
        "affect the Company's ability to meet customer demand and could harm "
        "its results of operations.\n\n"
        "Item 7. Management's Discussion and Analysis\n"
        "Total net sales | 383,285 | 394,328\n"
        "Net sales decreased 3 percent during fiscal 2023 compared to fiscal 2022, "
        "driven primarily by lower unit volumes partially offset by favourable mix.\n"
    )
