import time

import pandas as pd
from pandas.testing import assert_frame_equal

from utils.pipeline_utils import *

initialize()

input = list(range(1, 5))


@task
def loops(x, constant):
    time.sleep(0.1)
    return x + constant


@task
def loops_several_arguments(x, constant):
    time.sleep(0.1)
    return x + constant, x * constant


@filter_task
def df_filter(df):
    return df.iloc[-1:]


@filter_task
def list_filter(lst):
    new_list = []
    for i in range(len(lst)):
        if i % 2 == 0:
            new_list.append(i)
    return new_list


def test_task_exec():
    assert loops(1, 2).run() == 3


def test_task_sequential_exec():
    expected = [3, 4, 5, 6]

    assert seq_map(loops, input, constant=2).run() == expected


def test_task_parallel_exec():
    expected = [3, 4, 5, 6]

    assert par_map(loops, input, constant=2).run() == expected


def test_multiple_return_values_seq():
    x, y = map(list, zip(*seq_map(loops_several_arguments, input, constant=2).run()))

    assert x == [3, 4, 5, 6] and y == [2, 4, 6, 8]


def test_multiple_return_values_par():
    x, y = map(list, zip(*seq_map(loops_several_arguments, input, constant=2).run()))

    assert x == [3, 4, 5, 6] and y == [2, 4, 6, 8]


def test_df_filter():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    result = df_filter(df).run().reset_index(drop=True)
    expected = pd.DataFrame({"a": [5]}).reset_index(drop=True)
    assert_frame_equal(result, expected)


def test_list_filter():
    inp = list(range(0, 10))
    result = list_filter(inp).run()
    assert len(result) == 5
