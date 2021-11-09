import pandas as pd
from pandas.testing import assert_frame_equal

from rl.envs.utils.state_extender import StateExtenderCNN


def test_add_inventory_state():
    state = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = StateExtenderCNN().add_inventory_state(state, 99)
    expected = pd.DataFrame({"a": [1, 2, 3, 99], "b": [4, 5, 6, 99]}, index=[0, 1, 2, "inventory_state"])

    assert_frame_equal(result, expected)
