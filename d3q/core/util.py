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

import datetime
import os

import numpy as np
import tensorflow as tf
from d3q.core.logging import log
from gym.spaces import Space

DEFAULT_GAME = 'CartPole'


class Game:
    def __init__(self, game_name):
        import importlib

        import d3q.games as d3q_games_module

        self.MODULE_NAME = f'd3q.games.{game_name}'
        self.MODULE_FILE_PATH = f'{d3q_games_module.__path__._path[0]}/{game_name}.py'

        log.debug(f'Loading game module: {self.MODULE_NAME}')
        spec = importlib.util.spec_from_file_location(self.MODULE_NAME, self.MODULE_FILE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Set default values.
        self.GAME_NAME = game_name
        self.NUM_ENVS_PER_SIM = 1

        # Override values with the ones from the module.
        self.__dict__.update(module.__dict__)

    def make_env(self, *args, **kwargs):
        import gym
        return gym.make(self.GYM_NAME, *args, **kwargs)

    def preprocess_state(self, state):
        return state


def make_summary_writer(game: Game):
    tb_path = os.path.join('tblog', datetime.datetime.now().strftime(f"%Y%m%d-%H%M%S_{game.GAME_NAME}"))
    summary_writer = tf.summary.create_file_writer(tb_path)

    desc = [
        '| *Property* | *Value* |',
        '| ---------- | ------- |',
    ]

    for x, y in game.__dict__.items():
        if x[0].isupper():
            desc.append(f'| {x} | {y} |')

    with summary_writer.as_default(step=0):
        tf.summary.text("Training Session Description", '\n'.join(desc))

    return summary_writer


def make_sars_buffer_dtype(observation_space: Space, action_space: Space) -> np.dtype:
    return np.dtype([
        ('state', observation_space.dtype, observation_space.shape),
        ('action', action_space.dtype, action_space.shape),
        ('reward', np.float32, ()),
        ('state_next', observation_space.dtype, observation_space.shape),
        ('nonterminal', np.bool8, ()),
    ])


def cleanup_subprocesses_at_exit():
    def cleanup():
        import psutil
        current_process = psutil.Process()
        subprocesses = current_process.children(recursive=True)
        for subprocess in subprocesses:
            subprocess.kill()
        current_process.kill()

    import atexit
    atexit.register(cleanup)
