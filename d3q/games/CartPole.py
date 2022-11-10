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

from d3q.core.util import Game


class CartPoleGame(Game):
    def __init__(self):
        super().__init__('CartPole')

        self.GYM_NAME = 'CartPole-v1'

        self.NUM_SIMS = 16
        self.NUM_SKIP_FIRST_FRAMES = 5
        self.SKIP_FRAME_PROB = 0.0

        self.EXPERIENCE_SEND_BATCH_SIZE = 2 * 1024
        self.REPLAYMEMORY_CAPACITY = 256 * 1024
        self.FILL_REPLAYMEMORY_THRESHOLD = 32768
        self.LOSS_TO_PRIORITY_EXP_BETA = 0.0
        self.TERMINAL_PRIORITY_FACTOR = 3.0

        self.NUM_STEPS_PER_EPOCH = 1
        self.NUM_OPTIMIZATION_STEPS = 1
        self.NUM_OPTIMIZATION_SUBSTEPS = 1

        self.LOCAL_BATCH_SIZE = 36
        self.Q_GAMMA = 0.975
        # self.RANDOM_POLICY_THRESHOLD_DECAY = 0.98  <- Unused.

        self.OPTIMIZER = 'adam'
        self.LEARNING_RATE = 0.008
        self.CLIPVALUE = 1.0
        self.NUM_HIDDEN_UNITS = 32
        self.FINAL_ACTIVATION = 'linear'

        self.NUM_EVAL_ROUNDS = 1
        self.MAX_ROUND_STEPS = 5000
        self.GOAL_SCORE = 5000
        self.GAME_LIMIT_IN_SAMPLES = 2 * 1000 * 1000
        self.GAME_LIMIT_IN_SEC = 5 * 60

        self.PREVIEW_TILING = (2, 2)

    def make_loss_fn(self):
        import tensorflow as tf
        return tf.keras.losses.MeanSquaredError(reduction='none')

    def make_optimizer(self):
        import tensorflow as tf
        if self.OPTIMIZER == 'sgd':
            return tf.keras.optimizers.SGD(learning_rate=self.LEARNING_RATE, clipvalue=self.CLIPVALUE)
        elif self.OPTIMIZER == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE, clipvalue=self.CLIPVALUE)

    def make_model(self):
        import tensorflow as tf
        return tf.keras.Sequential([
            tf.keras.Input(shape=(4,), batch_size=self.LOCAL_BATCH_SIZE, dtype=tf.float32),

            tf.keras.layers.Normalization(
                mean=[[0.0, 0.0, 0.0, 0.0]],
                variance=[[0.00015, 0.005, 0.00015, 0.01]]),

            tf.keras.layers.Dense(self.NUM_HIDDEN_UNITS),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),

            tf.keras.layers.Dense(self.NUM_HIDDEN_UNITS),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),

            tf.keras.layers.Dense(2, activation=self.FINAL_ACTIVATION),
        ])

    def set_random_policy_threshold(self, old_random_policy_threshold: float):
        return (0.0, 1.0)

    def make_tune_space(self):
        from ray import tune
        return {
            'NUM_SKIP_FIRST_FRAMES': tune.randint(0, 5+1),
            'SKIP_FRAME_PROB': tune.uniform(0.0, 0.8),
            'REPLAYMEMORY_CAPACITY': tune.lograndint(16*1024, 1024*1024+1),
            'LOSS_TO_PRIORITY_EXP_BETA': tune.uniform(0.0, 1.0),
            'TERMINAL_PRIORITY_FACTOR': tune.uniform(1.0, 3.0),
            # 'NUM_STEPS_PER_EPOCH': tune.randint(1, 8+1),
            'NUM_OPTIMIZATION_STEPS': tune.randint(1, 128+1),
            # 'NUM_OPTIMIZATION_SUBSTEPS': tune.randint(1, 32+1),
            'LOCAL_BATCH_SIZE': tune.lograndint(32, 8*1024+1),
            'RANDOM_POLICY_THRESHOLD_DECAY': tune.loguniform(0.9, 0.99),
            'OPTIMIZER': tune.choice(['sgd', 'adam']),
            'LEARNING_RATE': tune.loguniform(0.0001, 0.01),
            'CLIPVALUE': tune.choice([None, 0.5, 1.0]),
            'NUM_HIDDEN_UNITS': tune.lograndint(4, 128+1),
            # 'FINAL_ACTIVATION': tune.choice(['linear', 'sigmoid', 'swish', 'tanh']),
        }
