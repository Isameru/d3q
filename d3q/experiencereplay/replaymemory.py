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
from d3q.core.logging import log
from numpy.random import default_rng

DEFAULT_MAX_BRANCH_BUCKET_CAPACITY = 256
DEFAULT_MAX_LEAF_BUCKET_CAPACITY = 16*1024


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

    @property
    def child_elem_count(self):
        assert self.children is not None
        return math.ceil((self.end_index - self.start_index) / len(self.children))

    def _verify(self):
        assert self.end_index > self.start_index
        assert (self.children is None) or (len(self.children) > 0)
        assert (self.parent is None) or (self.start_index >= self.parent.start_index and self.end_index <= self.parent.end_index)

    def update_priority_range(self, start_index: int, end_index: int, priorities):
        assert end_index > start_index, '...'
        assert priorities.shape == (end_index - start_index,), '...'
        assert start_index < self.end_index and end_index > self.start_index, '...'
        assert start_index >= self.start_index and end_index <= self.end_index, '...'
        if self.children is not None:
            child_elem_count = self.child_elem_count
            start_child_index = (start_index - self.start_index) // child_elem_count
            end_child_index = ((end_index - 1 - self.start_index) // child_elem_count) + 1

            cumulative_sum = self.cumulative_sums[start_child_index - 1] if start_child_index > 0 else 0.0
            for child_index in range(start_child_index, len(self.children)):
                if child_index < end_child_index:
                    child_start_data_index = max(start_index, self.start_index + child_elem_count * child_index)
                    child_end_data_index = min(end_index, self.start_index + child_elem_count * (child_index + 1))
                    cumulative_sum += self.children[child_index].update_priority_range(child_start_data_index,
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

    def update_indexed_priorities(self, indices, priorities):
        assert len(indices) > 0, 'Expected at least one priority to update.'
        assert len(indices) == len(priorities), 'Each passed index is expected to correspond to each of the priorities.'
        assert indices[0] >= self.start_index and indices[-1] < self.end_index, 'Only indices valid to this nodes are expected to be processed.'

        priority_offset = 0.0
        dirty_cumulated_sum_start_index = None

        if self.children is not None:
            child_elem_count = self.child_elem_count

            while len(indices) > 0:
                child_index = (indices[0] - self.start_index) // child_elem_count
                child_end_index = self.start_index + (child_index + 1) * child_elem_count

                indices_end_index = bisect_left(indices, child_end_index)
                assert indices_end_index > 0

                priority_at_child_index = self.cumulative_sums[child_index] - (self.cumulative_sums[child_index-1] if child_index > 0 else 0.0)

                if dirty_cumulated_sum_start_index is not None and child_index > dirty_cumulated_sum_start_index and priority_offset != 0.0:
                    self.cumulative_sums[dirty_cumulated_sum_start_index:child_index] += priority_offset

                new_children_priority_at_index = self.children[child_index].update_indexed_priorities(indices[:indices_end_index], priorities[:indices_end_index])
                priority_offset += new_children_priority_at_index - priority_at_child_index

                dirty_cumulated_sum_start_index = child_index

                indices = indices[indices_end_index:]
                priorities = priorities[indices_end_index:]

        else:
            indices = np.copy(indices) - self.start_index

            priority_offset = 0.0
            dirty_cumulated_sum_start_index = None

            for index, priority in zip(indices, priorities):
                priority_at_index = self.cumulative_sums[index] - (self.cumulative_sums[index-1] if index > 0 else 0.0)

                if dirty_cumulated_sum_start_index is not None and index > dirty_cumulated_sum_start_index and priority_offset != 0.0:
                    self.cumulative_sums[dirty_cumulated_sum_start_index:index] += priority_offset

                priority_offset += priority - priority_at_index

                dirty_cumulated_sum_start_index = index

        assert dirty_cumulated_sum_start_index is not None
        if priority_offset != 0.0:
            self.cumulative_sums[dirty_cumulated_sum_start_index:] += priority_offset

        return self.priority_sum

    def sample_indices_from_sorted_points(self, sample_points):
        sampled_indices = []
        sample_point_index = 0
        child_index = 0

        while sample_point_index < len(sample_points):
            recurse_child_index = bisect_left(self.cumulative_sums[child_index:], sample_points[sample_point_index]) + child_index
            if recurse_child_index < len(self.cumulative_sums):
                next_sample_point_index = bisect_left(sample_points[sample_point_index:], self.cumulative_sums[recurse_child_index]) + sample_point_index
            else:
                # It may happen due to cumulation of floating-point error.
                recurse_child_index = len(self.cumulative_sums) - 1
                next_sample_point_index = len(sample_points)

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
            bucket.cumulative_sums = np.zeros((len(bucket.children),), dtype=np.float64)
    if bucket.cumulative_sums is None:
        bucket.cumulative_sums = np.zeros((end_index-start_index,), dtype=np.float64)
    bucket._verify()
    return bucket


class ReplayMemory:
    def __init__(self,
                 capacity: int,
                 sars_dtype: np.dtype,
                 new_records_priority: float = None,
                 max_branch_bucket_capacity=DEFAULT_MAX_BRANCH_BUCKET_CAPACITY,
                 max_leaf_bucket_capacity=DEFAULT_MAX_LEAF_BUCKET_CAPACITY):
        self.capacity = capacity
        self.new_records_priority = new_records_priority

        log.info(f'Allocating buffer for prioritized experience replay of size: {self.capacity} × {sars_dtype.itemsize} bytes = {self.capacity * sars_dtype.itemsize} bytes')
        self.record_data = np.zeros(shape=(self.capacity,), dtype=sars_dtype)
        self.root_bucket = make_priority_tree(0, self.capacity, None, max_branch_bucket_capacity, max_leaf_bucket_capacity)

        self.size = 0
        self.next_free_index = 0
        self.virt_size = 0

    def memorize(self, records, priorities=None):
        assert len(records.shape) == 1 and records.dtype == self.record_data.dtype
        batch_size = records.shape[0]

        if priorities is None:
            priorities = np.full((len(records),), self.new_records_priority, np.float64)
        assert (priorities is not None) and all(priorities > 0.0), '...'

        log.debug(f'Memorizing {batch_size} new experiences, keeping {min(self.size + batch_size, self.capacity)} in total.')

        if batch_size > self.capacity:
            log.warning(f'Trying to memorize more experiences than the capacity of the replay memory (which is {self.capacity}). Truncating the memorized batch.')
            records = records[len(records)-self.capacity:]
            batch_size = self.capacity

        def push_data(dst_start_index: int, records, priorities):
            batch_size = records.shape[0]
            self.record_data[dst_start_index:dst_start_index + batch_size] = records
            self.root_bucket.update_priority_range(
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
        self.virt_size += batch_size

    def sample_random(self, batch_size: int, rng=None):
        if rng is None:
            rng = default_rng()
        sample_points = self.root_bucket.priority_sum * rng.random(batch_size)
        return self.sample_from_points(sample_points)

    def sample_from_points(self, sample_points, sample_points_sorted=False):
        if not sample_points_sorted:
            sample_points = np.sort(sample_points)
        indices = self.root_bucket.sample_indices_from_sorted_points(sample_points)
        virt_indices = np.array(indices, dtype=np.int64) + (self.virt_size - self.size)
        return np.take(self.record_data, indices, axis=0), virt_indices

    def update_priorities(self, sorted_virt_indices, priorities):
        assert sorted_virt_indices[-1] < self.virt_size

        unique_indices, unique_index_indices = np.unique(sorted_virt_indices, return_index=True)
        if len(unique_indices) != len(sorted_virt_indices):
            priorities = np.array([np.mean(priorities[unique_index_indices[i]:(unique_index_indices[i+1] if i < len(unique_index_indices)-1 else len(priorities))])
                                  for i in range(len(unique_indices))], dtype=np.float64)
            sorted_virt_indices = unique_indices

        sorted_indices = sorted_virt_indices - (self.virt_size - self.size)

        cut_to_index = bisect_left(sorted_indices, 0)
        sorted_indices = sorted_indices[cut_to_index:]
        priorities = priorities[cut_to_index:]

        if len(sorted_indices) == 0:
            return

        sorted_indices += self.next_free_index
        rotate_from_index = bisect_left(sorted_indices, self.size)

        if rotate_from_index < self.size:
            sorted_indices[rotate_from_index:] %= self.size
            sorted_indices = np.sort(sorted_indices)

        self.root_bucket.update_indexed_priorities(sorted_indices, priorities)
