import pandas as pd
from numpy import array_equal
from pandas.testing import assert_frame_equal

from rl.common.state_handler import StateHandlerCNN
from tests.utils import MockObj


def test_add_inventory_state():
    state = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = StateHandlerCNN(extend=True).extend_state(state, 99)
    expected = pd.DataFrame({"a": [1, 2, 3, 99], "b": [4, 5, 6, 99]}, index=[0, 1, 2, 0])

    assert_frame_equal(result, expected)


def test_forward():
    seq = MockObj(data=MockObj(arr=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})))
    shcnn = StateHandlerCNN(extend=True)

    result = shcnn.forward(seq, [99, 101])
    expected = pd.DataFrame({"a": [1, 2, 3, 99, 101], "b": [4, 5, 6, 99, 101]}, index=[0, 1, 2, 0, 0]).to_numpy()

    assert array_equal(result, expected)
