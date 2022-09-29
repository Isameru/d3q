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

import pickle
import random
from multiprocessing import Queue
from multiprocessing.queues import Empty

import numpy as np
import tensorflow as tf

from d3q.logging import configure_logger, log
from d3q.qlearn import evaluate
from d3q.util import make_game


class SimWorker:
    def __init__(self,
                 worker_name: str,
                 request_queue: Queue,
                 response_queue: Queue,
                 experience_queue: Queue,
                 game_name: str):
        configure_logger(logger_name=worker_name)

        self.worker_name = worker_name
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.experience_queue = experience_queue
        self.game = make_game(game_name)
        self.has_work = False

        self.env = None
        self.frame_count = None
        self.state0 = None  # The current state.

        # The following are created in reset_gym() based on gym environment specs.
        self.expecience_count: int = None
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.nonterminals = None

        self.model = self.game.make_model()
        self.random_policy_threshold = None

    def run(self):
        self.response_queue.put(('ready', None))

        while True:
            try:
                request_verb, request_data = self.request_queue.get(block=not self.has_work)
                getattr(self, request_verb)(*request_data)
            except Empty:
                pass
            if self.has_work:
                self.do_work_step()

    def noop(self):
        pass

    def reset_gym(self):
        if self.env is None:
            log.debug(f'Loading a gym: {self.game.GYM_NAME}')
            self.env = self.game.make_env()

            self.expecience_count = 0
            self.observations = np.empty(shape=[self.game.EXPERIENCE_SEND_BATCH_SIZE, *self.env.observation_space.shape], dtype=self.env.observation_space.dtype)
            assert self.env.action_space.dtype == np.int64
            self.actions = np.empty(shape=[self.game.EXPERIENCE_SEND_BATCH_SIZE, *self.env.action_space.shape], dtype=np.int8)
            self.rewards = np.empty(shape=[self.game.EXPERIENCE_SEND_BATCH_SIZE], dtype=np.float32)
            self.next_observations = np.empty(shape=[self.game.EXPERIENCE_SEND_BATCH_SIZE, *self.env.observation_space.shape], dtype=self.env.observation_space.dtype)
            self.nonterminals = np.empty(shape=[self.game.EXPERIENCE_SEND_BATCH_SIZE], dtype=np.int8)
        else:
            log.debug(f'Resetting the gym.')
        self.state0, _ = self.env.reset()
        self.frame_count = 0

    def set_target_network(self, model_params, start_sim: bool, random_policy_threshold: float):
        for tv, mp in zip(self.model.trainable_variables, model_params):
            tv.assign(mp)

        #self.random_policy_threshold = random_policy_threshold
        self.random_policy_threshold = random.uniform(0.0, 1.0)
        self.has_work = start_sim
        self.request_queue.put(('noop', ()))

        score = evaluate(self.game, self.model)
        self.response_queue.put(('evaluation', score), block=False)

    def do_work_step(self):
        if random.uniform(0.0, 1.0) < self.random_policy_threshold:
            action = self.env.action_space.sample()
        else:
            action_values = self.model(tf.expand_dims(self.state0, axis=0))
            action = tf.math.argmax(action_values, axis=1).numpy().item()

        state1, reward, terminal, _, _ = self.game.env_step(self.env, action)

        if self.frame_count > self.game.NUM_SKIP_FIRST_FRAMES and random.uniform(0.0, 1.0) > self.game.SKIP_FRAME_PROB:
            self.observations[self.expecience_count] = self.state0
            self.actions[self.expecience_count] = action
            self.rewards[self.expecience_count] = reward
            self.next_observations[self.expecience_count] = state1
            self.nonterminals[self.expecience_count] = 0 if terminal else 1
            self.expecience_count += 1

            if self.expecience_count == self.game.EXPERIENCE_SEND_BATCH_SIZE:
                log.debug(f'Sending a batch of {self.game.EXPERIENCE_SEND_BATCH_SIZE} experiences.')
                experiences = (np.copy(self.observations), np.copy(self.actions), np.copy(self.rewards), np.copy(self.next_observations), np.copy(self.nonterminals))
                self.experience_queue.put(pickle.dumps(experiences, protocol=pickle.HIGHEST_PROTOCOL), block=False)
                self.expecience_count = 0

        if terminal:
            self.state0, _ = self.env.reset()
            self.frame_count = 0
        else:
            self.state0 = state1
            self.frame_count += 1
