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

import random

import numpy as np
import tensorflow as tf

from d3q.logging import log

class ReplayMemory:
    def __init__(self, game):
        self.local_batch_size = game.LOCAL_BATCH_SIZE
        self.capacity = game.REPLAYMEMORY_CAPACITY
        self.size = 0

        env = game.make_env()
        with tf.device('/device:CPU:0'):
            self.observations = tf.Variable(
                tf.zeros([self.capacity, *env.observation_space.shape], dtype=env.observation_space.dtype),
                trainable=False, name='replaymemory/observations')
            assert env.action_space.dtype == np.int64
            self.actions = tf.Variable(
                tf.zeros([self.capacity, *env.action_space.shape], dtype=tf.int8),
                trainable=False, name='replaymemory/actions')
            self.rewards = tf.Variable(
                tf.zeros([self.capacity], dtype=tf.float32),
                trainable=False, name='replaymemory/rewards')
            self.next_observations = tf.Variable(
                tf.zeros([self.capacity, *env.observation_space.shape], dtype=env.observation_space.dtype),
                trainable=False, name='replaymemory/next_observations')
            self.nonterminals = tf.Variable(
                tf.zeros([self.capacity], dtype=tf.int8),
                trainable=False, name='replaymemory/nonterminals')
        self.next_free_index = 0
        del env

    def memorize(self, observation, action, reward, next_observation, nonterminal):
        log.info('Memorizing an experience.')
        self.observations[self.next_free_index].assign(observation)
        self.actions[self.next_free_index].assign(action)
        self.rewards[self.next_free_index].assign(reward)
        self.next_observations[self.next_free_index].assign(next_observation)
        self.nonterminals[self.next_free_index].assign(1 if nonterminal else 0)

        self.next_free_index += 1
        self.size = max(self.size, self.next_free_index)
        self.next_free_index = self.next_free_index % self.capacity

    def memorize_batch(self, observations, actions, rewards, next_observations, nonterminals):
        batch_size = observations.shape[0]
        log.info(f'Memorizing {batch_size} new experiences, keeping {min(self.size + batch_size, self.capacity)} in total.')
        if batch_size > self.capacity:
            log.warning(f'Trying to memorize more experiences than the capacity of the replay memory (which is {self.capacity}). Truncating the memorized batch. Consider reducing NUM_SIMS to receive less experiences.')
            observations = observations[0:self.capacity]
            actions = actions[0:self.capacity]
            rewards = rewards[0:self.capacity]
            next_observations = next_observations[0:self.capacity]
            nonterminals = nonterminals[0:self.capacity]
            batch_size = self.capacity

        def assign_batch(dst, src):
            assert dst.shape[1:] == src.shape[1:]
            assert batch_size == src.shape[0]
            free_left = self.capacity - self.next_free_index
            if batch_size <= free_left:
                dst[self.next_free_index:self.next_free_index+batch_size].assign(src)
            else:
                dst[self.next_free_index:self.capacity].assign(src[0:free_left])
                dst[0:batch_size-free_left].assign(src[free_left:])

        assign_batch(self.observations, observations)
        assign_batch(self.actions, actions)
        assign_batch(self.rewards, rewards)
        assign_batch(self.next_observations, next_observations)
        assign_batch(self.nonterminals, nonterminals)

        self.next_free_index += batch_size
        self.size = min(max(self.size, self.next_free_index), self.capacity)
        self.next_free_index = self.next_free_index % self.capacity

    def sample_random(self):
        indices = self._generate_random_indices(self.local_batch_size)
        return self._sample_indices(indices)

    def _generate_random_indices(self, batch_size: int):
        assert batch_size <= self.size, f"Cannot sample a batch of size {batch_size}, as there are only {self.size} elements in the replay buffer."
        return random.sample(range(0, self.size), batch_size)

    def _sample_indices(self, indices):
        return (
            tf.gather(self.observations, indices, validate_indices=False, axis=0, name='replaymemory/gather_observations'),
            tf.gather(self.actions, indices, validate_indices=False, axis=0, name='replaymemory/gather_actions'),
            tf.gather(self.rewards, indices, validate_indices=False, axis=0, name='replaymemory/gather_rewards'),
            tf.gather(self.next_observations, indices, validate_indices=False, axis=0, name='replaymemory/gather_next_observations'),
            tf.gather(self.nonterminals, indices, validate_indices=False, axis=0, name='replaymemory/gather_nonterminals')
        )
