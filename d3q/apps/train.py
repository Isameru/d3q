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
import warnings

with warnings.catch_warnings():
    try:
        import keras
    except ModuleNotFoundError:
        pass

from d3q.core.dqntrainer import DQNTrainer
from d3q.core.logging import DEFAULT_LOG_LEVEL, configure_logger, log
from d3q.core.util import (DEFAULT_GAME, make_game, make_sars_buffer_dtype,
                           make_summary_writer)
from d3q.experiencereplay.replaymemory_service import \
    ReplayMemoryServiceController
from d3q.sim.sim_service import SimPoolServiceController


def parse_args():
    """ Parses the command line.
    """
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', '--log-level', dest='log_level',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        default=DEFAULT_LOG_LEVEL,
                        help=f'console log level (default: {DEFAULT_LOG_LEVEL})')
    parser.add_argument('-g', '--game', dest='game',
                        default=DEFAULT_GAME,
                        help=f'name of supported AI Gym game (supported environment)')
    parser.add_argument('-m', '--model', dest='model',
                        default=os.path.join('models', 'net.tf'),
                        help=f'path to model (checkpoint)')
    parser.add_argument('-n', '--new-model', dest='new_model',
                        action='store_true',
                        default=False,
                        help=f'recreate the model from scratch rather than resuming from the checkpoint')
    parser.add_argument('-r', '--random-policy', dest='random_policy',
                        default=1.0, type=float,
                        help=f'initial random policy threshold (default: 1.0)')
    return parser.parse_args()


def main(args):
    replaymememory_srvc = None
    simpool_srvc = None

    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(3)

        configure_logger(log_level=args.log_level)

        # Instruct python.multiprocessing to spawn processes without forking, which shares the state of libraries with the parent process (like TensorFlow's global context).
        from multiprocessing import set_start_method
        set_start_method('spawn')

        # Instantiate the game object.
        game = make_game(args.game)

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

        # Load the existing model (if applicable).
        model_loaded = False
        if not args.new_model:
            try:
                model.load_weights(args.model)
                log.info(f'Model loaded: {args.model}')
                model_loaded = True
            except Exception:
                pass
        if not model_loaded:
            log.info(f'Creating a new model: {args.model}')

        summary_writer = make_summary_writer(game)

        # Use the Q-learning algorithm in an endless loop.
        qtrainer = DQNTrainer(game,
                              model,
                              args.model,
                              replaymememory_srvc,
                              simpool_srvc,
                              summary_writer,
                              args.random_policy)

        training_score = qtrainer.optimize_to_goal()
        log.info(f'Training Score: {training_score}')

    finally:
        if simpool_srvc is not None:
            simpool_srvc.close()
        if replaymememory_srvc is not None:
            replaymememory_srvc.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
