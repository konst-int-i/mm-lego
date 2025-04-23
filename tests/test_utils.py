import pytest
from mm_lego.utils import *
import time
import numpy as np

def test_config():
    # config smoke test
    config = Config("config/config_dev.yml").read()

    assert hasattr(config, "wandb")



@timed_function(unit="s")
def sleep_func(duration_s: float):
    time.sleep(duration_s)
    return 42


def test_timing_decorator():
    time, result = sleep_func(duration_s=0.1)
    np.testing.assert_almost_equal(time, 0.1, decimal=3)

    time, result = sleep_func(duration_s=0.2)
    np.testing.assert_almost_equal(time, 0.2, decimal=3)

    assert result == 42

