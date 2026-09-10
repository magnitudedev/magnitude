"""Device correctness work must not overlap a measured inference run."""

import pytest

from magnitude_engine.platform.measurement import exclusive_measurement


@pytest.fixture(scope="session", autouse=True)
def device_test_ownership():
    with exclusive_measurement():
        yield
