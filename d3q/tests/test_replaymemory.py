
import numpy as np
import pytest
from d3q.logging import configure_logger
from d3q.replaymemory2 import ReplayMemory

configure_logger(log_level='debug')


def test_priority_tree_1():
    rm = ReplayMemory(capacity=11,
                      record_size=4,
                      max_leaf_bucket_capacity=4,
                      max_branch_bucket_capacity=3)
    root = rm.root_bucket
    # | 0 0 0 0 | 0 0 0 0 | 0 0 0 |
    assert root.priority_sum == pytest.approx(0.0)

    root.update_priority(0, 11, np.full((11,), 1.0, dtype=np.float32))
    # | 1 1 1 1 | 1 1 1 1 | 1 1 1 |
    assert root.priority_sum == pytest.approx(11.0)

    root.update_priority(1, 3, np.full((2,), 2.0, dtype=np.float32))
    # | 1 2 2 1 | 1 1 1 1 | 1 1 1 |
    assert root.priority_sum == pytest.approx(13.0)

    root.update_priority(2, 10, np.full((8,), 3.0, dtype=np.float32))
    # | 1 2 3 3 | 3 3 3 3 | 3 3 1 |
    assert root.priority_sum == pytest.approx(28.0)

    root.update_priority(3, 6, np.full((3,), 4.0, dtype=np.float32))
    # | 1 2 3 4 | 4 4 3 3 | 3 3 1 |
    assert root.priority_sum == pytest.approx(31.0)

    root.update_priority(6, 9, np.full((3,), 4.0, dtype=np.float32))
    # | 1 2 3 4 | 4 4 4 4 | 4 3 1 |
    assert root.priority_sum == pytest.approx(34.0)

    root.update_priority(1, 2, np.full((1,), 5.0, dtype=np.float32))
    # | 1 5 3 4 | 4 4 4 4 | 4 3 1 |
    assert root.priority_sum == pytest.approx(37.0)

    root.update_priority(5, 7, np.full((2,), 6.0, dtype=np.float32))
    # | 1 5 3 4 | 4 6 6 4 | 4 3 1 |
    assert root.priority_sum == pytest.approx(41.0)

    root.update_priority(0, 11, np.full((11,), 1.0, dtype=np.float32))
    # | 1 1 1 1 | 1 1 1 1 | 1 1 1 |
    assert root.priority_sum == pytest.approx(11.0)

    root.update_priority(0, 11, np.full((11,), 0.0, dtype=np.float32))
    # | 0 0 0 0 | 0 0 0 0 | 0 0 0 |
    assert root.priority_sum == pytest.approx(0.0)


def test_priority_tree_2():
    for max_leaf_bucket_capacity, max_branch_bucket_capacity in [(11, 7), (37, 41), (101, 79), (244, 563)]:
        rm = ReplayMemory(capacity=10001,
                          record_size=4,
                          max_leaf_bucket_capacity=max_leaf_bucket_capacity,
                          max_branch_bucket_capacity=max_branch_bucket_capacity)
        root = rm.root_bucket
        assert root.priority_sum == pytest.approx(0.0)

        root.update_priority(0, 10001, np.full((10001,), 1.0, dtype=np.float32))
        assert root.priority_sum == pytest.approx(10001.0)

        root.update_priority(0, 10001, np.full((10001,), 2.0, dtype=np.float32))
        assert root.priority_sum == pytest.approx(20002.0)

        start_index = 1234
        len = 4567
        root.update_priority(start_index, start_index+len, np.full((len,), 4.0, dtype=np.float32))
        assert root.priority_sum == pytest.approx(20002.0 + 2.0*len)


def test_priority_tree_3():
    rm = ReplayMemory(capacity=1000,
                      record_size=4,
                      max_leaf_bucket_capacity=25,
                      max_branch_bucket_capacity=32)
    root = rm.root_bucket
    assert root.priority_sum == pytest.approx(0.0)

    t = np.cumsum(np.full((1000,), 0.01, dtype=np.float32))
    s = np.sum(t)

    root.update_priority(0, 1000, t)
    assert root.priority_sum == pytest.approx(s)


def test_sample():
    rm = ReplayMemory(capacity=64,
                      record_size=4,
                      max_leaf_bucket_capacity=3,
                      max_branch_bucket_capacity=4)
    for i in range(64):
        eb = np.zeros((1, 4), dtype=np.ubyte)
        eb[0] = np.frombuffer((i).to_bytes(4, 'big'), dtype=np.ubyte)
        rm.memorize(eb, priorities=np.array([10*i], dtype=np.float32))

    x = rm.sample_random(128)
    pass

    # rm = ReplayMemory(capacity=1000,
    #                   record_size=4,
    #                   max_leaf_bucket_capacity=32,
    #                   max_branch_bucket_capacity=8)
    # for i in range(1000):
    #     eb = np.zeros((1, 4), dtype=np.ubyte)
    #     eb[0] = np.frombuffer((i).to_bytes(4, 'big'), dtype=np.ubyte)
    #     rm.memorize(eb, np.array([(i + 500) / 1000.0], dtype=np.float32))

    # rm.sample_random(32)

# for i in range(100):
# eb = np.zeros((3, 4), dtype=np.ubyte)
# eb[0] = np.frombuffer((3*i).to_bytes(4, 'big'), dtype=np.ubyte)
# eb[1] = np.frombuffer((3*i+1).to_bytes(4, 'big'), dtype=np.ubyte)
# eb[2] = np.frombuffer((3*i+2).to_bytes(4, 'big'), dtype=np.ubyte)
# rm.memorize(eb, 1.0)
# rm.memorize(eb, 2.0)
# rm.memorize(eb, 3.0)
# rm.memorize(eb, 4.0)
