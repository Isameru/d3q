# autopep8: off
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

import math
from bisect import bisect_left

import numpy as np
from numpy.random import default_rng

from d3q.core.logging import log

DEFAULT_MAX_BRANCH_BUCKET_CAPACITY = 32
DEFAULT_MAX_LEAF_BUCKET_CAPACITY = 128


class PriorityTreeNode:
    def __init__(self, start_index: int, end_index: int, parent=None):
        self.start_index = start_index
        self.end_index = end_index
        self.parent = parent
        self.children = None
        self.cumulative_sums = None

    @property
    def priority_sum(self):
        return self.cumulative_sums[-1]

    def _verify(self):
        assert self.end_index > self.start_index
        assert (self.children is None) or (len(self.children) > 0)
        assert (self.parent is None) or (self.start_index >= self.parent.start_index and self.end_index <= self.parent.end_index)

    def update_priority(self, start_index: int, end_index: int, priorities):
        assert end_index > start_index, '...'
        assert priorities.shape == (end_index - start_index,), '...'
        assert start_index < self.end_index and end_index > self.start_index, '...'
        assert start_index >= self.start_index and end_index <= self.end_index, '...'
        if self.children is not None:
            child_elem_count = math.ceil((self.end_index - self.start_index) / len(self.children))
            start_child_index = (start_index - self.start_index) // child_elem_count
            end_child_index = ((end_index - 1 - self.start_index) // child_elem_count) + 1

            cumulative_sum = self.cumulative_sums[start_child_index - 1] if start_child_index > 0 else 0.0
            for child_index in range(start_child_index, len(self.children)):
                if child_index < end_child_index:
                    child_start_data_index = max(start_index, self.start_index + child_elem_count * child_index)
                    child_end_data_index = min(end_index, self.start_index + child_elem_count * (child_index + 1))
                    cumulative_sum += self.children[child_index].update_priority(child_start_data_index,
                                                                                 child_end_data_index,
                                                                                 priorities[child_start_data_index-start_index:child_end_data_index-start_index])
                else:
                    cumulative_sum += self.cumulative_sums[child_index] - previous_child_cumulative_sum
                previous_child_cumulative_sum = self.cumulative_sums[child_index]
                self.cumulative_sums[child_index] = cumulative_sum
        else:
            pre_priority_offset = self.cumulative_sums[start_index - 1 - self.start_index] if (start_index > self.start_index) else 0.0
            post_priority_offset = -(self.cumulative_sums[end_index - 1 - self.start_index]) if (end_index < self.end_index) else None
            self.cumulative_sums[start_index-self.start_index:end_index-self.start_index] = np.cumsum(priorities) + pre_priority_offset
            if end_index < self.end_index:
                post_priority_offset += (self.cumulative_sums[end_index-self.start_index-1]) if (end_index < self.end_index) else 0.0
                self.cumulative_sums[end_index-self.start_index:] += post_priority_offset
        return self.cumulative_sums[-1]

    def sample_indices_from_sorted_points(self, sample_points):
        sampled_indices = []
        sample_point_index = 0
        child_index = 0

        while sample_point_index < len(sample_points):
            recurse_child_index = bisect_left(self.cumulative_sums[child_index:], sample_points[sample_point_index]) + child_index
            next_sample_point_index = bisect_left(sample_points[sample_point_index:], self.cumulative_sums[recurse_child_index]) + sample_point_index

            if self.children is not None:
                child_cumulative_sum_offset = -self.cumulative_sums[recurse_child_index-1] if recurse_child_index > 0 else 0.0
                sampled_indices.extend(self.children[recurse_child_index].sample_indices_from_sorted_points(sample_points[sample_point_index:next_sample_point_index] + child_cumulative_sum_offset))
            else:
                sampled_indices.extend([self.start_index + recurse_child_index] * (next_sample_point_index - sample_point_index))

            sample_point_index = next_sample_point_index
            child_index = recurse_child_index + 1

        assert len(sampled_indices) == len(sample_points)
        return sampled_indices


def make_priority_tree(start_index: int,
                       end_index: int,
                       parent: PriorityTreeNode,
                       max_branch_bucket_capacity: int,
                       max_leaf_bucket_capacity: int):
    elem_count = end_index - start_index
    bucket = PriorityTreeNode(start_index, end_index, parent)
    if elem_count > max_leaf_bucket_capacity:
        branch_elem_count = math.ceil(elem_count / max_branch_bucket_capacity)
        branch_count = math.ceil(elem_count / branch_elem_count)
        if branch_count > 1:
            bucket.children = [
                make_priority_tree(
                    start_index + branch_index * branch_elem_count,
                    min(end_index, start_index + (branch_index + 1) * branch_elem_count),
                    bucket,
                    max_branch_bucket_capacity,
                    max_leaf_bucket_capacity)
                for branch_index in range(branch_count)]
            bucket.cumulative_sums = np.zeros((len(bucket.children),), dtype=np.float32)
    if bucket.cumulative_sums is None:
        bucket.cumulative_sums = np.zeros((end_index-start_index,), dtype=np.float32)
    bucket._verify()
    return bucket


class ReplayMemory:
    def __init__(self,
                 capacity: int,
                 sars_dtype: np.dtype,
                 initial_records_priority: float = None,
                 max_branch_bucket_capacity=DEFAULT_MAX_BRANCH_BUCKET_CAPACITY,
                 max_leaf_bucket_capacity=DEFAULT_MAX_LEAF_BUCKET_CAPACITY):
        self.capacity = capacity
        self.initial_records_priority = initial_records_priority

        log.info(f'Allocating buffer for prioritized experience replay of size: {self.capacity} × {sars_dtype.itemsize} bytes = {self.capacity * sars_dtype.itemsize} bytes')
        self.record_data = np.zeros(shape=(self.capacity,), dtype=sars_dtype)
        self.root_bucket = make_priority_tree(0, self.capacity, None, max_branch_bucket_capacity, max_leaf_bucket_capacity)

        self.size = 0
        self.next_free_index = 0

    def memorize(self, records, priorities=None):
        assert len(records.shape) == 1 and records.dtype == self.record_data.dtype
        batch_size = records.shape[0]

        if priorities is None:
            priorities = np.full((len(records),), self.initial_records_priority, np.float32)
        assert (priorities is not None) and all(priorities > 0.0), '...'

        log.info(f'Memorizing {batch_size} new experiences, keeping {min(self.size + batch_size, self.capacity)} in total.')

        if batch_size > self.capacity:
            log.warning(
                f'Trying to memorize more experiences than the capacity of the replay memory (which is {self.capacity}). Truncating the memorized batch.')
            records = records[len(records)-self.capacity:]
            batch_size = self.capacity

        def push_data(dst_start_index: int, records, priorities):
            batch_size = records.shape[0]
            self.record_data[dst_start_index:dst_start_index + batch_size] = records
            self.root_bucket.update_priority(
                dst_start_index,
                dst_start_index+batch_size,
                priorities)

        free_left = self.capacity - self.next_free_index
        if batch_size <= free_left:
            push_data(self.next_free_index, records, priorities)
        else:
            push_data(self.next_free_index, records[0:free_left], priorities[0:free_left])
            push_data(0, records[free_left:], priorities[free_left:])

        self.next_free_index += batch_size
        self.size = min(max(self.size, self.next_free_index), self.capacity)
        self.next_free_index = self.next_free_index % self.capacity

    def sample_random(self, batch_size: int, rng=None):
        if rng is None:
            rng = default_rng()
        sample_points = self.root_bucket.priority_sum * rng.random(batch_size)
        return self.sample_from_points(sample_points)

    def sample_from_points(self, sample_points):
        indices = self.root_bucket.sample_indices_from_sorted_points(np.sort(sample_points))
        return np.take(self.record_data, indices, axis=0)

    # TODO: Add virtual indices and means to update priorities.
