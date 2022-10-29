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

import os
import time

import cv2 as cv
import numpy as np
import tensorflow as tf
from d3q.core.logging import DEFAULT_LOG_LEVEL, configure_logger, log
from d3q.core.util import DEFAULT_GAME, make_game

MODEL_CHECK_INTERVAL_SEC = 2.5
OVERLAY_SHOW_FOR_STEPS = 50


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
                        default='models\\net.tf',
                        help=f'path to model (checkpoint)')
    return parser.parse_args()


class ModelSource:
    def __init__(self, game, model_path):
        self.game = game
        self.model_path = model_path

        self.last_time_checked = None
        self.model = None
        self.model_ts = None
        self.model_ver = 0

    def get_model(self):
        if self.model is None:
            self.model = self.game.make_model()

        if (self.last_time_checked is None) or (time.time() - self.last_time_checked > MODEL_CHECK_INTERVAL_SEC):
            self.last_time_checked = time.time()
            try:
                model_file_ts = os.path.getmtime(f'{self.model_path}.index')
                if self.model_ts is None or model_file_ts > self.model_ts:
                    with open(self.model_path, 'w'):
                        self.model.load_weights(self.model_path)
                    self.model_ts = model_file_ts
                    self.model_ver += 1
            except Exception as err:
                log.warning(f'Failed to load model: {err}')
                self.model = None
                self.model_ts = None
                # `self.model_ver` remains as it is.

        return self.model, self.model_ver


class EnvPlayer:
    def __init__(self, game, env, render_target, model_source):
        self.game = game
        self.env = env
        self.render_target = render_target
        self.model_source = model_source

        self.model = None
        self.model_ver_text = None
        self.state0 = None
        self.iter_num = None
        self.score = None

        self.overlay_show_countdown = 0
        self.overlay_title = None
        self.overlay_title_color = None

    def step(self):
        if self.overlay_show_countdown == 0:
            if self.state0 is None:
                self.state0, _ = self.env.reset()
                self.iter_num = 0
                self.score = 0.0
            else:
                if self.model is None:
                    self.model, model_ver = self.model_source.get_model()
                    self.model_ver_text = str(model_ver) if self.model is not None else 'RANDOM'

                if self.model is not None:
                    action_values = self.model(tf.expand_dims(self.state0, axis=0)).numpy()
                    action = np.argmax(action_values)
                else:
                    action = self.env.action_space.sample()

                state1, reward, terminal, _, _ = self.game.env_step(self.env, action)

                self.iter_num += 1
                self.score += reward

                if terminal or self.iter_num > self.game.MAX_ROUND_STEPS:
                    self.model = None
                    self.state0 = None
                    self.overlay_show_countdown = OVERLAY_SHOW_FOR_STEPS
                    self.overlay_title = 'GAME OVER'
                    self.overlay_title_color = (32, 32, 32)
                else:
                    self.state0 = state1

        self.render_target[:] = self.env.render()

        cv.putText(self.render_target, 'MODEL VER.', (0, 12), cv.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1, cv.LINE_AA)
        cv.putText(self.render_target, f'{self.model_ver_text}', (86, 12), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 128, 0), 1, cv.LINE_AA)
        cv.putText(self.render_target, 'SCORE', (0, 24), cv.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1, cv.LINE_AA)
        cv.putText(self.render_target, f'{self.score:0.2f}', (86, 24), cv.FONT_HERSHEY_SIMPLEX, 0.4, (112, 25, 25), 1, cv.LINE_AA)

        if self.overlay_show_countdown > 0:
            self.render_target[:] = self.render_target // 6 * 5
            self.overlay_show_countdown -= 1

            text_size = cv.getTextSize(self.overlay_title, cv.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            cv.putText(self.render_target, self.overlay_title, ((self.render_target.shape[0] - text_size[1]) // 2,
                       (self.render_target.shape[1] - text_size[0]) // 2), cv.FONT_HERSHEY_SIMPLEX, 1.0, self.overlay_title_color, 2, cv.LINE_AA)


def main(args):
    game = make_game(args.game)
    model_source = ModelSource(game, args.model)

    # Make multiple environments based on the game's desired preview tiling.
    num_envs = np.prod(game.PREVIEW_TILING)
    envs = [game.make_env(render_mode='rgb_array') for _ in range(num_envs)]

    # Obtain the rendering's dimensions and prepare the tiled render target.
    some_env = envs[0]
    some_env.reset()
    some_rendering = some_env.render()

    render_target = np.zeros(shape=(some_rendering.shape[0] * game.PREVIEW_TILING[0], some_rendering.shape[1] * game.PREVIEW_TILING[1], some_rendering.shape[2]), dtype=some_rendering.dtype)
    window_name = f'{game.GAME_NAME} :: d3q Preview'
    cv.namedWindow(window_name)

    # Create multiple environment controllers (players).
    env_players = []
    for env_index, env in enumerate(envs):
        lu_pos = ((env_index // game.PREVIEW_TILING[1]) * some_rendering.shape[0], (env_index % game.PREVIEW_TILING[1]) * some_rendering.shape[1])
        render_target_view = render_target[lu_pos[0]:lu_pos[0]+some_rendering.shape[0], lu_pos[1]:lu_pos[1]+some_rendering.shape[1]]
        env_players.append(EnvPlayer(game, env, render_target_view, model_source))

    while True:
        for env_player in env_players:
            env_player.step()

        cv.imshow(window_name, render_target)
        key_pressed = cv.waitKey(1)
        if key_pressed == 27:
            break


if __name__ == '__main__':
    args = parse_args()
    configure_logger(log_level=args.log_level)
    main(args)
