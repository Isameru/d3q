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

from math import prod

import numpy as np
from d3q.core.logging import log
from gym.spaces import Space

DEFAULT_GAME = 'CartPole'


class Game:
    pass


def make_game(game_name: str):
    import importlib

    import d3q.games as d3q_games_module

    module_name = f'd3q.games.{game_name}'
    file_path = f'{d3q_games_module.__path__._path[0]}/{game_name}.py'

    log.info(f'Loading game module: {module_name}')

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    game = Game()
    game.GAME_NAME = game_name
    game.__dict__.update(module.__dict__)

    return game


def make_sars_buffer_dtype(observation_space: Space, action_space: Space) -> np.dtype:
    return np.dtype([
        ('state', observation_space.dtype, observation_space.shape),
        ('action', action_space.dtype, action_space.shape),
        ('reward', np.float32, ()),
        ('state_next', observation_space.dtype, observation_space.shape),
        ('nonterminal', np.bool8, ()),
    ])
