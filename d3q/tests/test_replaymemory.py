#
#         .o8    .oooo.
#        "888  .dP""Y88b
#    .oooo888        ]8P'  .ooooo oo
#   d88' `888      <88b.  d88' `888
#   888   888       `88b. 888   888
#   888   888  o.   .88P  888   888
#   `Y8bod88P" `8bd88P'   `V8bod888
#                               888.
#                               8P'
#                               "

import numpy as np
import pytest
from d3q.core.logging import configure_logger
from d3q.experiencereplay.replaymemory import ReplayMemory

configure_logger(log_level='debug')


def test_priority_tree_1():
    rm = ReplayMemory(capacity=11,
                      sars_dtype=np.dtype(np.int32),
                      max_leaf_bucket_capacity=4,
                      max_branch_bucket_capacity=3)
    root = rm.root_bucket
    # | 0 0 0 0 | 0 0 0 0 | 0 0 0 |
    assert root.priority_sum == pytest.approx(0.0)

    root.update_priority_range(0, 11, np.full((11,), 1.0, dtype=np.float64))
    # | 1 1 1 1 | 1 1 1 1 | 1 1 1 |
    assert root.priority_sum == pytest.approx(11.0)

    root.update_priority_range(1, 3, np.full((2,), 2.0, dtype=np.float64))
    # | 1 2 2 1 | 1 1 1 1 | 1 1 1 |
    assert root.priority_sum == pytest.approx(13.0)

    root.update_priority_range(2, 10, np.full((8,), 3.0, dtype=np.float64))
    # | 1 2 3 3 | 3 3 3 3 | 3 3 1 |
    assert root.priority_sum == pytest.approx(28.0)

    root.update_priority_range(3, 6, np.full((3,), 4.0, dtype=np.float64))
    # | 1 2 3 4 | 4 4 3 3 | 3 3 1 |
    assert root.priority_sum == pytest.approx(31.0)

    root.update_priority_range(6, 9, np.full((3,), 4.0, dtype=np.float64))
    # | 1 2 3 4 | 4 4 4 4 | 4 3 1 |
    assert root.priority_sum == pytest.approx(34.0)

    root.update_priority_range(1, 2, np.full((1,), 5.0, dtype=np.float64))
    # | 1 5 3 4 | 4 4 4 4 | 4 3 1 |
    assert root.priority_sum == pytest.approx(37.0)

    root.update_priority_range(5, 7, np.full((2,), 6.0, dtype=np.float64))
    # | 1 5 3 4 | 4 6 6 4 | 4 3 1 |
    assert root.priority_sum == pytest.approx(41.0)

    root.update_priority_range(0, 11, np.full((11,), 1.0, dtype=np.float64))
    # | 1 1 1 1 | 1 1 1 1 | 1 1 1 |
    assert root.priority_sum == pytest.approx(11.0)

    root.update_priority_range(0, 11, np.full((11,), 0.0, dtype=np.float64))
    # | 0 0 0 0 | 0 0 0 0 | 0 0 0 |
    assert root.priority_sum == pytest.approx(0.0)


def test_priority_tree_2():
    for max_leaf_bucket_capacity, max_branch_bucket_capacity in [(11, 7), (37, 41), (101, 79), (244, 563)]:
        rm = ReplayMemory(capacity=10001,
                          sars_dtype=np.dtype(np.int32),
                          max_leaf_bucket_capacity=max_leaf_bucket_capacity,
                          max_branch_bucket_capacity=max_branch_bucket_capacity)
        root = rm.root_bucket
        assert root.priority_sum == pytest.approx(0.0)

        root.update_priority_range(0, 10001, np.full((10001,), 1.0, dtype=np.float64))
        assert root.priority_sum == pytest.approx(10001.0)

        root.update_priority_range(0, 10001, np.full((10001,), 2.0, dtype=np.float64))
        assert root.priority_sum == pytest.approx(20002.0)

        start_index = 1234
        len = 4567
        root.update_priority_range(start_index, start_index+len, np.full((len,), 4.0, dtype=np.float64))
        assert root.priority_sum == pytest.approx(20002.0 + 2.0*len)


def test_priority_tree_3():
    rm = ReplayMemory(capacity=1000,
                      sars_dtype=np.dtype(np.int32),
                      max_leaf_bucket_capacity=25,
                      max_branch_bucket_capacity=32)
    root = rm.root_bucket
    assert root.priority_sum == pytest.approx(0.0)

    t = np.cumsum(np.full((1000,), 0.01, dtype=np.float64))
    s = np.sum(t)

    root.update_priority_range(0, 1000, t)
    assert root.priority_sum == pytest.approx(s)


def test_sample_1():
    rm = ReplayMemory(capacity=64,
                      sars_dtype=np.dtype(np.int32),
                      max_leaf_bucket_capacity=3,
                      max_branch_bucket_capacity=4)
    for i in range(64):
        eb = np.array([i], dtype=np.int32)
        rm.memorize(eb, priorities=np.array([10.0*i + 0.1], dtype=np.float64))

    rm.sample_random(128)


def test_sample_2():
    rm = ReplayMemory(capacity=10,
                      sars_dtype=np.dtype(np.int32),
                      max_leaf_bucket_capacity=2,
                      max_branch_bucket_capacity=2)

    eb = np.arange(0, 10, 1, dtype=np.int32)
    rm.memorize(eb, priorities=np.full(shape=(10,), fill_value=1.0, dtype=np.float64))

    sp = np.arange(0.0, 10.0, 0.01, dtype=np.float64)
    recs, virt_indices = rm.sample_from_points(sp, sample_points_sorted=True)

    assert (np.unique(recs) == eb).all()


def test_update_prio_1():
    rm = ReplayMemory(capacity=14,
                      sars_dtype=np.dtype(np.int32),
                      max_leaf_bucket_capacity=4,
                      max_branch_bucket_capacity=3)
    eb = np.arange(0, 14, 1, dtype=np.int32)
    rm.memorize(eb, priorities=np.full(shape=(14,), fill_value=1.0, dtype=np.float64))
    # | 1 1 1 1 1 | 1 1 1 1 1 | 1 1 1 1 |
    assert rm.root_bucket.priority_sum == pytest.approx(14.0)

    sp = np.arange(0.5, 14, 1.0, dtype=np.float64)
    sp = np.random.permutation(sp)
    recs, virt_indices = rm.sample_from_points(sp)
    assert (recs == eb).all()
    assert (virt_indices == eb).all()

    rm.update_priorities(np.array([1, 3, 5, 7, 9, 11, 13], dtype=np.int64), np.full((7,), 2.0, dtype=np.float64))
    # | 1 2 1 2 1 | 2 1 2 1 2 | 1 2 1 2 |
    assert rm.root_bucket.priority_sum == pytest.approx(21.0)

    rm.update_priorities(np.array([2, 6, 12], dtype=np.int64), np.full((3,), 3.0, dtype=np.float64))
    # | 1 2 3 2 1 | 2 3 2 1 2 | 1 2 3 2 |
    assert rm.root_bucket.priority_sum == pytest.approx(27.0)

    rm.update_priorities(np.array([0, 10], dtype=np.int64), np.full((2,), 0.0, dtype=np.float64))
    # | 0 2 3 2 1 | 2 3 2 1 2 | 0 2 3 2 |
    assert rm.root_bucket.priority_sum == pytest.approx(25.0)


def test_update_prio_2():
    rm = ReplayMemory(capacity=10,
                      sars_dtype=np.dtype(np.int32),
                      max_leaf_bucket_capacity=10,
                      max_branch_bucket_capacity=1)
    eb = np.arange(0, 10, 1, dtype=np.int32)
    rm.memorize(eb, priorities=np.full(shape=(10,), fill_value=1.0, dtype=np.float64))
    rm.memorize(eb, priorities=np.full(shape=(10,), fill_value=1.0, dtype=np.float64))
    # | 1 1 1 1 1 1 1 1 1 1 |
    assert rm.root_bucket.priority_sum == pytest.approx(10.0)

    sp = np.arange(0.5, 10, 1.0, dtype=np.float64)
    sp = np.random.permutation(sp)
    recs, virt_indices = rm.sample_from_points(sp)
    assert (recs == eb).all()
    assert (virt_indices == (eb + 10)).all()

    eb = np.arange(10, 14, 1, dtype=np.int32)
    rm.memorize(eb, priorities=np.full(shape=(4,), fill_value=2.0, dtype=np.float64))
    # | 2 2 2 2 1 1 1 1 1 1 |
    assert rm.root_bucket.priority_sum == pytest.approx(14.0)

    sp = np.arange(0.0, rm.root_bucket.priority_sum, 0.07, dtype=np.float64)
    recs, virt_indices = rm.sample_from_points(sp, sample_points_sorted=True)

    assert min(recs) == 4
    assert max(recs) == 13
    assert min(virt_indices) == 14
    assert max(virt_indices) == 23

    eb = np.arange(14, 17, 1, dtype=np.int32)
    rm.memorize(eb, priorities=np.full(shape=(3,), fill_value=3.0, dtype=np.float64))
    # | 2 2 2 2 3 3 3 1 1 1 |
    assert rm.root_bucket.priority_sum == pytest.approx(20.0)

    rm.update_priorities(virt_indices, priorities=np.full(virt_indices.shape, fill_value=4.0, dtype=np.float64))
    # | 4 4 4 4 3 3 3 4 4 4 |
    assert rm.root_bucket.priority_sum == pytest.approx(37.0)
