from copy import deepcopy

import pytest

from utils.util_funcs import *


def test_update_check_key():
    d = {
        "a": 10,
        "b": 11
    }

    result = update_check_key(deepcopy(d), {"a": 99})
    assert result == {"a": 99, "b": 11}

    with pytest.raises(ValueError):
        update_check_key(deepcopy(d), {"c": 99})
