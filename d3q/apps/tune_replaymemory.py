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

import sys
import traceback

from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch

CAPACITY = 512*1024
BATCH_SIZE = 8*1024
NUM_STEPS = 128

NUM_TRIALS = 1024
MAX_CONCURRENT = 8


def benchmark_replaymemory(config):
    try:
        import time

        import numpy as np
        from d3q.core.logging import configure_logger
        from d3q.experiencereplay.replaymemory import ReplayMemory

        configure_logger(log_level='error')
        rm = ReplayMemory(capacity=CAPACITY,
                          sars_dtype=np.dtype(np.int32),
                          max_leaf_bucket_capacity=config['max_leaf_bucket_capacity'],
                          max_branch_bucket_capacity=config['max_branch_bucket_capacity'])

        batch = np.arange(0, BATCH_SIZE, 1, dtype=np.int32)
        priorities = np.full(shape=(BATCH_SIZE,), fill_value=1.0, dtype=np.float64)

        start_time = time.time()
        for step in range(NUM_STEPS):
            rm.memorize(batch, priorities)
            _, virt_indices = rm.sample_random(BATCH_SIZE)
            rm.update_priorities(virt_indices, priorities)

        end_time = time.time()
        tune.report(duration=end_time - start_time)
    except Exception as err:
        print(f'error: While benchmarking variant {config}: {str(err)}')
        traceback.print_exception(*sys.exc_info())
        tune.report(duration=999999)


space = {
    "max_branch_bucket_capacity": tune.lograndint(2**1, 2**16+1),
    "max_leaf_bucket_capacity": tune.lograndint(2**1, 2**16+1),
}

search_alg = HyperOptSearch(metric="duration", mode="min")
search_alg = ConcurrencyLimiter(search_alg, max_concurrent=MAX_CONCURRENT)
tune_config = tune.TuneConfig(num_samples=NUM_TRIALS,
                              search_alg=search_alg)
tuner = tune.Tuner(benchmark_replaymemory,
                   param_space=space,
                   tune_config=tune_config)

results = tuner.fit()

print(results.get_best_result(metric="duration", mode="min").config)
