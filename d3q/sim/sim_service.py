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
from multiprocessing import Process, Queue
from multiprocessing.queues import Empty

import gym
import numpy as np
import tensorflow as tf
from d3q.core.logging import configure_logger, log
from d3q.core.util import make_game, make_sars_buffer_dtype


class SimPoolServiceController:
    def __init__(self, game, experience_queues):
        self.game = game
        self.num_sims = game.NUM_SIMS
        assert len(experience_queues) == self.num_sims
        self.controllers = [SimServiceController(f'sim:{i}', game.GAME_NAME, experience_queues[i]) for i in range(self.num_sims)]
        for controller in self.controllers:
            controller.expect_ready()
        log.info('All simulators ready.')

    def _request_async(self, verb, data):
        for controller in self.controllers:
            controller.request_queue.put((verb, data,))

    def _get_responses(self, expected_verb):
        return [controller.get_response(expected_verb) for controller in self.controllers]

    def set_target_network(self,
                           model: tf.keras.Model,
                           start_sim: bool,
                           random_policy_threshold: float):
        log.debug(f'Evaluating a new target model by running {self.game.NUM_EVAL_ROUNDS} rounds on {self.num_sims} workers, each round of at most {self.game.MAX_ROUND_STEPS} steps.')

        model_params = [tv.numpy() for tv in model.trainable_variables]
        self._request_async('set_target_network', (model_params, start_sim, random_policy_threshold))

        if start_sim:
            log.debug(f'Running simulations on the target model with random_policy_threshold: {random_policy_threshold}')

    def receive_evalution_scores(self):
        scores = self._get_responses('evaluation')
        log.info(f'Best Score: {max(scores)} | Average Score: {sum(scores)/len(scores)}')
        return scores


class SimServiceController:
    def __init__(self, name: str, game_name: str, experience_queue: Queue):
        self.request_queue = Queue()
        self.response_queue = Queue()
        self.process = Process(target=sim_serviceprocessor_fn,
                               args=(name,
                                     game_name,
                                     self.request_queue,
                                     self.response_queue,
                                     experience_queue))
        self.process.start()

    def expect_ready(self):
        assert self.response_queue.get(block=True)[0] == 'ready'

    def get_response(self, expected_verb):
        verb, data = self.response_queue.get(block=True)
        assert verb == expected_verb, f'Expected response verb "{expected_verb}", but got "{verb}".'
        return data

    def request_async(self, verb, data):
        self.request_queue.put((verb, data), block=False)


def sim_serviceprocessor_fn(*args, **kwargs):
    processor = SimServiceProcessor(*args, **kwargs)
    processor.run()


class SimServiceProcessor:
    def __init__(self,
                 worker_name: str,
                 game_name: str,
                 request_queue: Queue,
                 response_queue: Queue,
                 experience_queue: Queue):
        configure_logger(logger_name=worker_name)

        self.worker_name = worker_name
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.experience_queue = experience_queue

        self.game = make_game(game_name)
        self.has_work = False

        log.debug(f'Loading a gym: {self.game.GYM_NAME}')
        self.use_async_envs = (self.game.NUM_ENVS_PER_SIM > 1)
        if self.use_async_envs:
            self.async_envs = gym.vector.AsyncVectorEnv([self.game.make_env] * self.game.NUM_ENVS_PER_SIM)
            self.state0_vec, _ = self.async_envs.reset()
            self.frame_counts = [0] * self.game.NUM_ENVS_PER_SIM
            sars_dtype = make_sars_buffer_dtype(self.async_envs.observation_space, self.async_envs.action_space)
        else:
            self.env = self.game.make_env()
            self.state0, _ = self.env.reset()
            self.frame_count: int = 0
            sars_dtype = make_sars_buffer_dtype(self.env.observation_space, self.env.action_space)

        self.experience_count: int = 0
        self.sars_buffer = np.empty(shape=(self.game.EXPERIENCE_SEND_BATCH_SIZE,), dtype=sars_dtype)

        self.model = self.game.make_model()
        self.random_policy_threshold = None

    def run(self):
        self.response_queue.put(('ready', None))

        while True:
            try:
                while self.request_queue.qsize() > 0:
                    request_verb, request_data = self.request_queue.get(block=not self.has_work)
                    getattr(self, request_verb)(*request_data)
            except Empty:
                pass

            if self.has_work and (self.experience_queue.qsize() < 1):
                if self.use_async_envs:
                    self.do_work_step_with_async_envs()
                else:
                    self.do_work_step_with_env()

    def noop(self):
        pass

    def set_target_network(self,
                           model_params,
                           start_sim: bool,
                           random_policy_threshold: float):
        for tv, mp in zip(self.model.trainable_variables, model_params):
            tv.assign(mp)

        # TODO: Implement proper random policy strategy.
        self.random_policy_threshold = random_policy_threshold
        self.has_work = start_sim
        self.request_queue.put(('noop', ()))

        score = self.evaluate()
        self.response_queue.put(('evaluation', score), block=False)

    def do_work_step_with_env(self):
        if random.uniform(0.0, 1.0) < self.random_policy_threshold:
            action = self.env.action_space.sample()
        else:
            action_values = self.model(tf.expand_dims(self.state0, axis=0))
            action = tf.math.argmax(action_values, axis=1).numpy().item()

        state1, reward, terminal, _, _ = self.game.env_step(self.env, action)

        if self.frame_count == self.game.MAX_ROUND_STEPS:
            terminal = True

        if terminal or (self.frame_count > self.game.NUM_SKIP_FIRST_FRAMES and random.uniform(0.0, 1.0) > self.game.SKIP_FRAME_PROB):
            self.submit_sars(self.state0, action, reward, state1, 0 if terminal else 1)

        if terminal:
            self.state0, _ = self.env.reset()
            self.frame_count = 0
        else:
            self.state0 = state1
            self.frame_count += 1

    def do_work_step_with_async_envs(self):
        if random.uniform(0.0, 1.0) < self.random_policy_threshold:
            action_vec = self.async_envs.action_space.sample()
        else:
            action_values_vec = self.model(self.state0_vec)
            action_vec = tf.math.argmax(action_values_vec, axis=1).numpy()

        state1_vec, reward_vec, terminal_vec, _, info = self.async_envs.step(action_vec)
        state_terminal_vec = info.get('final_observation', None)

        raise NotImplementedError('Only NUM_ENVS_PER_SIM=1 is supported')

    def submit_sars(self, *args):
        self.sars_buffer[self.experience_count] = args
        self.experience_count += 1

        if self.experience_count == self.game.EXPERIENCE_SEND_BATCH_SIZE:
            log.debug(f'Sending a batch of {self.game.EXPERIENCE_SEND_BATCH_SIZE} experiences.')
            self.experience_queue.put(pickle.dumps(self.sars_buffer, protocol=pickle.HIGHEST_PROTOCOL), block=False)
            self.experience_count = 0

    def evaluate(self):
        score = 0.0

        for round in range(self.game.NUM_EVAL_ROUNDS):
            reward_sum = 0.0
            state0, _ = self.env.reset()
            for step in range(self.game.MAX_ROUND_STEPS):
                action_values = self.model(tf.expand_dims(state0, axis=0)).numpy()
                action = np.argmax(action_values)
                state1, reward, terminal, info, _ = self.game.env_step(self.env, action)
                reward_sum += reward
                if terminal:
                    break
                else:
                    state0 = state1
            score += reward_sum

        if self.game.NUM_EVAL_ROUNDS > 0:
            score /= self.game.NUM_EVAL_ROUNDS

        self.state0, _ = self.env.reset()
        self.frame_count = 0

        return score
