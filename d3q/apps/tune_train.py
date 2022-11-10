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

import os
import sys
import traceback
import warnings

from d3q.core.logging import configure_logger, log
from d3q.core.util import DEFAULT_GAME, make_game
from ray import air, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch

NUM_TRIALS = 1024


def benchmark_training(config):
    replaymememory_srvc = None
    simpool_srvc = None

    try:
        with warnings.catch_warnings():
            try:
                import keras
            except ModuleNotFoundError:
                pass

        from d3q.core.dqntrainer import DQNTrainer
        from d3q.core.util import make_sars_buffer_dtype, make_summary_writer
        from d3q.experiencereplay.replaymemory_service import \
            ReplayMemoryServiceController
        from d3q.sim.sim_service import SimPoolServiceController

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(3)
        configure_logger(log_level='info')

        # Instruct python.multiprocessing to spawn processes without forking, which shares the state of libraries with the parent process (like TensorFlow's global context).
        from multiprocessing import set_start_method
        set_start_method('spawn')

        # Instantiate the game object and apply the config.
        game = make_game(args.game)
        game.update_from_config(config)

        # Retrieve the gym environment's observation and action spaces.
        sample_env = game.make_env()
        env_observation_space = sample_env.observation_space
        env_action_space = sample_env.action_space
        del sample_env

        # Create the replay memory buffer as a service running asynchronously in a separate process.
        sars_buffer_dtype = make_sars_buffer_dtype(env_observation_space, env_action_space)
        replaymememory_srvc = ReplayMemoryServiceController(game, sars_buffer_dtype)

        # Create a pool of simulator workers as a service running asynchronously in multiple separate processes.
        simpool_srvc = SimPoolServiceController(game, replaymememory_srvc.experience_queues)

        # Instantiate the TensorFlow model used for training.
        model = game.make_model()

        summary_writer = make_summary_writer(game)
        # Use the Q-learning algorithm in an endless loop.
        qtrainer = DQNTrainer(game,
                              model,
                              None,
                              replaymememory_srvc,
                              simpool_srvc,
                              summary_writer,
                              1.0)

        training_score = qtrainer.optimize_to_goal()
        tune.report(score=training_score)

    except Exception as err:
        print(f'error: While benchmarking variant {config}: {str(err)}')
        traceback.print_exception(*sys.exc_info())
        tune.report(score=0.0)

    finally:
        if simpool_srvc is not None:
            simpool_srvc.close()
        if replaymememory_srvc is not None:
            replaymememory_srvc.close()


def parse_args():
    """ Parses the command line.
    """
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-g', '--game', dest='game',
                        default=DEFAULT_GAME,
                        help=f'name of supported AI Gym game (supported environment)')
    parser.add_argument('-m', '--model', dest='model',
                        default=os.path.join('models', 'net.tf'),
                        help=f'path to model (checkpoint)')
    parser.add_argument('-n', '--name', dest='name',
                        required=True,
                        help=f'name of ray.tune session (directory)')
    return parser.parse_args()


def main(args):
    configure_logger(log_level='info')

    # Instantiate the game object to retrieve the tune search space.
    game = make_game(args.game)
    tune_space = game.make_tune_space()

    num_trials = NUM_TRIALS

    search_alg = HyperOptSearch(metric="score", mode="max")
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=1)
    tune_config = tune.TuneConfig(num_samples=num_trials,
                                  search_alg=search_alg)

    raytune_path = os.path.abspath(os.path.join('ray_results', args.name))
    if os.path.isdir(raytune_path):
        log.info(f'Restoring tuner from: {raytune_path}')
        tuner = tune.Tuner.restore(raytune_path, resume_unfinished=True, restart_errored=True)
    else:
        log.info(f'Starting a new tuner: {raytune_path}')
        tuner = tune.Tuner(benchmark_training,
                           param_space=tune_space,
                           tune_config=tune_config,
                           run_config=air.RunConfig(local_dir="ray_results", name=args.name))

    results = tuner.fit()

    print(results.get_best_result(metric="score", mode="max").config)


if __name__ == '__main__':
    args = parse_args()
    main(args)
