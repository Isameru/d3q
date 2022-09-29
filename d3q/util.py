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
from functools import lru_cache

from d3q.logging import log

DEFAULT_GAME = 'CartPole'


@lru_cache(1)
def use_hpu():
    return os.environ['TRAIN_DEVICE'].lower() == 'hpu'


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
    game.__dict__.update(module.__dict__)

    return game
