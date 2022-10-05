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

import tensorflow as tf

GYM_NAME = 'CartPole-v1'
NUM_SIMS = 8
NUM_SKIP_FIRST_FRAMES = 3
SKIP_FRAME_PROB = 0.75
EXPERIENCE_SEND_BATCH_SIZE = 1024
REPLAYMEMORY_CAPACITY = 128 * 1024
FILL_REPLAYMEMORY_THRESHOLD = REPLAYMEMORY_CAPACITY // 4
LOCAL_BATCH_SIZE = 512
Q_GAMMA = 0.975
NUM_OPTIMIZATION_STEPS_PER_ITER = 32
NUM_OPTIMIZATION_SUBSTEPS = 1
NUM_ITERS_PER_CHECKPOINT = 4
RANDOM_POLICY_THRESHOLD_DECAY = 0.99
NUM_EVAL_ROUNDS = 1
MAX_ROUND_STEPS = 1000
GOAL_SCORE = 1000
PREVIEW_TILING=(2, 2)

def make_loss_fn():
    return tf.keras.losses.MeanSquaredError()

def make_optimizer():
    # return tf.keras.optimizers.Adam(learning_rate=0.0003, clipvalue=1.0)
    return tf.keras.optimizers.SGD(learning_rate=0.001, clipvalue=1.0)

def make_model():
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

def make_env(*args, **kwargs):
    import gym
    return gym.make(GYM_NAME, *args, **kwargs)

def env_step(env, action):
    state, reward, terminal, info, info2 = env.step(action)
    return state, reward, terminal, info, info2
