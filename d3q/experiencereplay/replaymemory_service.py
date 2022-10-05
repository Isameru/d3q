# autopep8: on
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
from multiprocessing import Process, Queue
from multiprocessing.queues import Empty

import numpy as np
import tensorflow as tf
from d3q.core.logging import configure_logger
from d3q.core.util import make_game
from d3q.experiencereplay.replaymemory import ReplayMemory
from numpy.random import default_rng


class ReplayMemoryServiceController:
    def __init__(self,
                 game,
                 sars_dtype):
        self.game = game

        self.request_queue = Queue()
        self.response_queue = Queue()
        self.experience_queues = [Queue() for _ in range(self.game.NUM_SIMS)]
        self.memory_queue = Queue()

        self.process = Process(target=replaymemory_serviceprocessor_fn,
                               args=(game.GAME_NAME,
                                     sars_dtype,
                                     self.request_queue,
                                     self.response_queue,
                                     self.experience_queues,
                                     self.memory_queue))
        self.process.start()
        self._expect_ready()

    def _expect_ready(self):
        assert self.response_queue.get(block=True)[0] == 'ready'

    def fetch_sampled_memories(self, as_tf_tensor_on_device=None):
        memories = pickle.loads(self.memory_queue.get(block=True))
        memories = list(memories[part_name] for part_name in ['state', 'action', 'reward', 'state_next', 'nonterminal'])

        if as_tf_tensor_on_device is not None:
            with tf.device(as_tf_tensor_on_device):
                memories = [tf.constant(memories_part) for memories_part in memories]
        return memories


def replaymemory_serviceprocessor_fn(*args, **kwargs):
    processor = ReplayMemoryServiceProcessor(*args, **kwargs)
    processor.run()


class ReplayMemoryServiceProcessor:
    def __init__(self,
                 game_name,
                 sars_dtype,
                 request_queue,
                 response_queue,
                 experience_queues,
                 memory_queue):
        configure_logger(logger_name='repmem')
        self.game = make_game(game_name)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.experience_queues = experience_queues
        self.memory_queue = memory_queue

        self.replaymemory = ReplayMemory(self.game.REPLAYMEMORY_CAPACITY, sars_dtype)
        self.rng = default_rng()

    def run(self):
        self.response_queue.put(('ready', None))

        while True:
            # Process the incoming requests first.
            try:
                while self.request_queue.qsize() > 0:
                    request_verb, request_data = self.request_queue.get(block=False)
                    getattr(self, request_verb)(*request_data)
            except Empty:
                pass

            # TODO: Add logic which chooses which is more important: memorizing new experiencing or providing memory samples to the trainer.

            # Fetch and memorize experiences.
            experience_records_vec = []
            for experience_queue in self.experience_queues:
                try:
                    while experience_queue.qsize() > 0:
                        experience_records_vec.append(pickle.loads(experience_queue.get(block=False)))
                except Empty:
                    pass
            if len(experience_records_vec) > 0:
                experience_records = np.concatenate(experience_records_vec, axis=0)
                priorities = np.full(shape=(experience_records.shape[0],), fill_value=1.0, dtype=np.float32)
                self.replaymemory.memorize(experience_records, priorities)

            if self.memory_queue.qsize() < 1 and self.replaymemory.size >= self.game.FILL_REPLAYMEMORY_THRESHOLD:
                memories = self.replaymemory.sample_random(self.game.LOCAL_BATCH_SIZE, self.rng)
                self.memory_queue.put(pickle.dumps(memories, protocol=pickle.HIGHEST_PROTOCOL), block=False)
