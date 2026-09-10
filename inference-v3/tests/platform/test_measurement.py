"""Whole-run ownership includes loading and can contain component measurements."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from magnitude_engine.platform.measurement import exclusive_measurement


def claim():
    with exclusive_measurement():
        return True


def test_measurement_nested_owner_does_not_admit_peer_threads_or_processes():
    source = (
        "from magnitude_engine.platform.measurement import exclusive_measurement\n"
        "with exclusive_measurement(): pass"
    )
    with exclusive_measurement():
        with exclusive_measurement():
            with ThreadPoolExecutor(max_workers=1) as pool:
                with pytest.raises(RuntimeError, match="another thread"):
                    pool.submit(claim).result()
            result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
            assert result.returncode != 0
            assert "another inference benchmark" in result.stderr
        # Leaving a nested scope must not release the outer process lock.
        result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
        assert result.returncode != 0
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
