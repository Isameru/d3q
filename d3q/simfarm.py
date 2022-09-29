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
from multiprocessing import Process, Queue, set_start_method
from multiprocessing.queues import Empty

import tensorflow as tf

from d3q.logging import log


class SimProcessHandler:
    def __init__(self, name: str, game_name: str):
        self.request_queue = Queue()
        self.response_queue = Queue()
        self.experience_queue = Queue()
        self.process = Process(target=sim_main_fn,
                               args=(name, self.request_queue, self.response_queue, self.experience_queue, game_name))
        self.process.start()

    def expect_ready(self):
        assert self.response_queue.get(block=True)[0] == 'ready'

    def get_response(self, expected_verb):
        verb, data = self.response_queue.get(block=True)
        assert verb == expected_verb, f'Expected response verb "{expected_verb}", but got "{verb}".'
        return data

    def request_async(self, verb, data):
        self.request_queue.put((verb, data), block=False)

    def fetch_experiences(self):
        try:
            while True:
                yield pickle.loads(self.experience_queue.get(block=False))
        except Empty:
            pass


class SimFarm:
    def __init__(self, game):
        set_start_method('spawn')
        self.game = game
        self.num_sims = game.NUM_SIMS
        self.process_handlers = [SimProcessHandler(f'sim:{i}', game.GAME_NAME) for i in range(self.num_sims)]
        for ph in self.process_handlers:
            ph.expect_ready()
        log.info('All simulation workers ready.')

    def _request_async(self, verb, data):
        for ph in self.process_handlers:
            ph.request_queue.put((verb, data,))

    def _get_responses(self, expected_verb):
        return [ph.get_response(expected_verb) for ph in self.process_handlers]

    def reset_gym(self):
        self._request_async('reset_gym', ())

    def set_target_network(self, model: tf.keras.Model, start_sim: bool, random_policy_threshold: float):
        log.info(f'Evaluating a new target model by running {self.game.NUM_EVAL_ROUNDS} rounds on {self.num_sims} workers, each round of at most {self.game.MAX_EVAL_ROUND_STEPS} steps.')

        model_params = [tv.numpy() for tv in model.trainable_variables]
        self._request_async('set_target_network', (model_params, start_sim, random_policy_threshold))

        scores = self._get_responses('evaluation')
        log.info(f'Best Score: {max(scores)} | Average Score: {sum(scores)/len(scores)}')
        if start_sim:
            log.info(f'Running simulations on the target model with random_policy_threshold: {random_policy_threshold}')
        return scores

    def fetch_experiences(self):
        experiences = []
        for ph in self.process_handlers:
            experiences.extend(ph.fetch_experiences())

        if len(experiences) == 0:
            return None

        observation_batches, action_batches, reward_batches, next_observation_batch, nonterminal_batches = zip(*experiences)

        return (
            tf.concat(observation_batches, axis=0, name='fetch_experiences/concat_observation_batches'),
            tf.concat(action_batches, axis=0, name='fetch_experiences/concat_action_batches'),
            tf.concat(reward_batches, axis=0, name='fetch_experiences/concat_reward_batches'),
            tf.concat(next_observation_batch, axis=0, name='fetch_experiences/concat_next_observation_batch'),
            tf.concat(nonterminal_batches, axis=0, name='fetch_experiences/concat_nonterminal_batches')
        )


def sim_main_fn(name: str, request_queue: Queue, response_queue: Queue, experience_queue: Queue, game_name: str):
    import os
    os.environ = {}
    from d3q.simworker import SimWorker
    simworker = SimWorker(name, request_queue, response_queue, experience_queue, game_name)
    simworker.run()
