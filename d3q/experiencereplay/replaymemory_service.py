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
import time
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
                               args=(game.config,
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
        memories, virt_indices, sampling_saturation = pickle.loads(self.memory_queue.get(block=True))
        memories = list(memories[part_name] for part_name in ['state', 'action', 'reward', 'state_next', 'nonterminal'])

        if as_tf_tensor_on_device is not None:
            with tf.device(as_tf_tensor_on_device):
                memories = [tf.convert_to_tensor(memories_part) for memories_part in memories]
        return *memories, virt_indices, sampling_saturation

    def update_priorities(self, sorted_virt_indices, priorities):
        self.request_queue.put(('update_priorities', pickle.dumps((sorted_virt_indices, priorities), protocol=pickle.HIGHEST_PROTOCOL)), block=False)

    def close(self):
        self.request_queue.put(('close', pickle.dumps((), protocol=pickle.HIGHEST_PROTOCOL)))
        try:
            self.process.join(timeout=3.0)
        except TimeoutError:
            self.process.terminate()
        self.request_queue.close()
        self.response_queue.close()
        for queue in self.experience_queues:
            queue.close()
        self.memory_queue.close()


def replaymemory_serviceprocessor_fn(*args, **kwargs):
    processor = ReplayMemoryServiceProcessor(*args, **kwargs)
    processor.run()


class ReplayMemoryServiceProcessor:
    def __init__(self,
                 game_config,
                 sars_dtype,
                 request_queue,
                 response_queue,
                 experience_queues,
                 memory_queue):
        configure_logger(logger_name='repmem')

        self.request_queue = request_queue
        self.response_queue = response_queue
        self.experience_queues = experience_queues
        self.memory_queue = memory_queue

        self.game = make_game(game_config['GAME_NAME'])
        self.game.update_from_config(game_config)

        self.should_exit = False

        self.replaymemory = ReplayMemory(self.game.REPLAYMEMORY_CAPACITY, sars_dtype)
        self.rng = default_rng()

        self.sampling_saturation = 0.0

    def run(self):
        self.response_queue.put(('ready', None))

        while True:
            # Process the incoming requests first.
            try:
                while self.request_queue.qsize() > 0:
                    request_verb, request_data = self.request_queue.get(block=False)
                    getattr(self, request_verb)(*pickle.loads(request_data))
                    if self.should_exit:
                        return
            except Empty:
                pass

            # TODO: Determine whether the sampling saturation should play a role in a logic which chooses which is more important: memorizing new experiencing or providing memory samples to the trainer.
            action_taken = False

            if self.memory_queue.qsize() < 1 and \
                    self.replaymemory.size >= min(self.game.FILL_REPLAYMEMORY_THRESHOLD, self.replaymemory.capacity):

                memories, virt_indices = self.replaymemory.sample_random(self.game.LOCAL_BATCH_SIZE, self.rng)
                self.memory_queue.put(pickle.dumps((memories, virt_indices, self.sampling_saturation), protocol=pickle.HIGHEST_PROTOCOL), block=False)

                self.sampling_saturation += self.game.LOCAL_BATCH_SIZE / self.replaymemory.size
                action_taken = True

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

                if self.replaymemory.size > 0:
                    new_experiences_priority = self.replaymemory.root_bucket.priority_sum / self.replaymemory.size
                else:
                    new_experiences_priority = 1.0
                priorities = np.full(shape=(experience_records.shape[0],), fill_value=new_experiences_priority, dtype=np.float64)

                if self.game.TERMINAL_PRIORITY_FACTOR != 1.0:
                    priorities += priorities * (self.game.TERMINAL_PRIORITY_FACTOR - 1.0) * (1 - experience_records['nonterminal'].view(np.uint8))

                self.replaymemory.memorize(experience_records, priorities)

                self.sampling_saturation *= 1.0 - self.game.LOCAL_BATCH_SIZE / self.replaymemory.size
                action_taken = True

            if not action_taken:
                time.sleep(0.001)

    def update_priorities(self, sorted_virt_indices, priorities):
        self.replaymemory.update_priorities(sorted_virt_indices, priorities)

    def close(self):
        self.request_queue.close()
        self.response_queue.close()
        for queue in self.experience_queues:
            queue.close()
        self.memory_queue.close()
        self.should_exit = True
