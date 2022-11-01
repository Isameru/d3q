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

GYM_NAME = 'CartPole-v1'

NUM_SIMS = 16
NUM_SKIP_FIRST_FRAMES = 3
SKIP_FRAME_PROB = 0.7

EXPERIENCE_SEND_BATCH_SIZE = 1024
REPLAYMEMORY_CAPACITY = 128 * 1024
FILL_REPLAYMEMORY_THRESHOLD = REPLAYMEMORY_CAPACITY // 4
LOSS_TO_PRIORITY_EXP_BETA = 0.7
TERMINAL_PRIORITY_FACTOR = 1.5

NUM_STEPS_PER_EPOCH = 8
NUM_OPTIMIZATIONS_PER_STEP = 64

LOCAL_BATCH_SIZE = 1024
Q_GAMMA = 0.975
#RANDOM_POLICY_THRESHOLD_DECAY = 0.99

NUM_EVAL_ROUNDS = 1
MAX_ROUND_STEPS = 5000
GOAL_SCORE = 5000
GAME_LIMIT_IN_SAMPLES = 16 * 1000 * 1000

PREVIEW_TILING = (2, 2)


def make_loss_fn():
    import tensorflow as tf
    return tf.keras.losses.MeanSquaredError(reduction='none')


def make_optimizer():
    import tensorflow as tf
    return tf.keras.optimizers.SGD(learning_rate=0.001, clipvalue=1.0)


def make_model():
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.Input(shape=(4,), batch_size=LOCAL_BATCH_SIZE, dtype=tf.float32),

        tf.keras.layers.Normalization(
            mean=[[0.0, 0.0, 0.0, 0.0]],
            variance=[[0.00015, 0.005, 0.00015, 0.01]]),

        tf.keras.layers.Dense(16),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(tf.nn.relu),

        tf.keras.layers.Dense(16),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(tf.nn.relu),

        tf.keras.layers.Dense(2),
    ])


def set_random_policy_threshold(old_random_policy_threshold: float):
    import random
    return random.uniform(0.0, 1.0)
    # return old_random_policy_threshold * RANDOM_POLICY_THRESHOLD_DECAY


def make_tune_space():
    from ray import tune
    return {
        "NUM_SKIP_FIRST_FRAMES": tune.choice([0, 1, 2, 3, 4, 5]),
        "SKIP_FRAME_PROB": tune.uniform(0.0, 0.8),
        "REPLAYMEMORY_CAPACITY": tune.choice([16*1024, 32*1024, 64*1024, 128*1024, 256*1024, 512*1024, 1024*1024]),
        "LOSS_TO_PRIORITY_EXP_BETA": tune.uniform(0.0, 2.0),
        "TERMINAL_PRIORITY_FACTOR": tune.uniform(1.0, 3.0),
        "LOCAL_BATCH_SIZE": tune.choice([32, 64, 128, 256, 512, 1024, 2*1024, 4*1024, 8*1024]),
    }
