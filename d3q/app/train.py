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
    try:
        import keras
    except ModuleNotFoundError:
        pass

from d3q.logging import DEFAULT_LOG_LEVEL, configure_logger, log
from d3q.qlearn import QTrainer
from d3q.replaymemory import ReplayMemory
from d3q.simfarm import SimFarm
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
        help=f'name of supported AI Gym game (supported environment)')
    parser.add_argument('-m', '--model', dest='model',
        default='models/net.tf',
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
    game = make_game(args.game)

    model = game.make_model()

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

    simfarm = SimFarm(game)
    simfarm.reset_gym()

    replaymemory = ReplayMemory(game)
    qtrainer = QTrainer(game, model, replaymemory)

    iter_num = 0
    train_steps_since_last_sync = None
    random_policy_threshold = args.random_policy

    for iter_num, _ in enumerate(iter(bool, True), start=0):
        if iter_num == 0 or train_steps_since_last_sync >= game.NUM_ITERS_PER_CHECKPOINT:
            simfarm.set_target_network(model, True, random_policy_threshold)
            train_steps_since_last_sync = 0
            random_policy_threshold *= game.RANDOM_POLICY_THRESHOLD_DECAY

            # if rank == 0:
            #     log.info(f'Saving model: {args.model}')
            #     model.save_weights(args.model)
            #     comm.barrier()
            #     comm.barrier()
            # else:
            #     comm.barrier()
            #     model.load_weights(args.model)
            #     comm.barrier()
            log.info(f'Saving model: {args.model}')
            model.save_weights(args.model)

        experiences = simfarm.fetch_experiences()
        if experiences is not None:
            replaymemory.memorize_batch(*experiences)

        if replaymemory.size > game.FILL_REPLAYMEMORY_THRESHOLD:
            qtrainer.optimize()
            train_steps_since_last_sync += 1

if __name__ == '__main__':
    args = parse_args()
    configure_logger(log_level=args.log_level)
    main(args)
