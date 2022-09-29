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

import warnings

with warnings.catch_warnings():
    import keras

import time

import numpy as np
import tensorflow as tf
from d3q.logging import DEFAULT_LOG_LEVEL, configure_logger, log
from d3q.util import DEFAULT_GAME, make_game


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
        help=f'name of supported AI Gym game (environment)')
    parser.add_argument('-m', '--model', dest='model',
        default='models/net.tf',
        help=f'path to model (checkpoint)')
    return parser.parse_args()


def main(args):
    game = make_game(args.game)
    env = game.make_env(render_mode='human')
    model = game.make_model()

    while True:
        try:
            model.load_weights(args.model)
            log.info(f'Model loaded: {args.model}')
        except Exception:
            log.info('Waiting for the model...')
            time.sleep(2.5)
            continue

        state0, _ = env.reset()
        score = 0.0
        for iter_num, _ in enumerate(iter(bool, True), start=0):
            # env.render(mode='rgb_array')
            env.render()
            action_values = model(tf.expand_dims(state0, axis=0)).numpy()
            print(action_values)
            action = np.argmax(action_values)
            state1, reward, terminal, info, _ = game.env_step(env, action)
            score += reward
            if terminal:
                break
            else:
                state0 = state1
        log.info(f'Scoring {score} after {iter_num} steps.')

if __name__ == '__main__':
    args = parse_args()
    configure_logger(log_level=args.log_level)
    main(args)
